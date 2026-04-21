"""
engine/retrieval_eval.py
========================
Retrieval-stage evaluation module.

Khớp với interface của ``agent/main_agent.py``:
  agent.query(question) → {
      "answer": str,
      "retrieved_ids": List[str],   # ← dùng field này
      "metadata": {...}
  }

Hai nguồn retrieved_ids được hỗ trợ:
1. **Từ agent response trực tiếp** (runner gọi inline sau mỗi query).
2. **Từ trường tĩnh trong dataset** (``retrieved_ids`` đã lưu sẵn trong golden_set.jsonl).

Metrics
-------
- Hit Rate @ k  : 1.0 nếu ≥1 expected_id có trong top-k retrieved, else 0.0.
- MRR           : 1 / rank của expected_id đầu tiên tìm thấy (0.0 nếu không có).

evaluate_batch() trả về per-case lẫn aggregate, phân biệt rõ:
- ``evaluated``   : case tính được cả hai metric.
- ``skipped``     : case thiếu ít nhất một trong hai trường bắt buộc.
- ``no_gt``       : case có retrieved_ids nhưng không có expected_retrieval_ids
                    (không thể tính metric, ghi nhận riêng để debug).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Đánh giá chất lượng Retrieval stage độc lập với Generation."""

    # ------------------------------------------------------------------ #
    #  Per-case metrics                                                    #
    # ------------------------------------------------------------------ #

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> float:
        """
        Hit Rate @ k
        ============
        Trả về 1.0 nếu ít nhất 1 ``expected_ids`` xuất hiện trong ``top_k``
        kết quả đầu tiên của ``retrieved_ids``; 0.0 ngược lại.

        Parameters
        ----------
        expected_ids   : Ground-Truth doc IDs (từ dataset hoặc human annotation).
        retrieved_ids  : Doc IDs do agent.retrieve() trả về theo thứ tự rank.
                         Khớp với ``response["retrieved_ids"]`` của MainAgent.
        top_k          : Cắt kết quả ở top-k (mặc định 3, khớp MainAgent.top_k).

        Returns
        -------
        float : 1.0 (hit) hoặc 0.0 (miss).
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
        expected_set = set(expected_ids)
        top_k_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in expected_set for doc_id in top_k_retrieved) else 0.0

    def calculate_mrr(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
    ) -> float:
        """
        Mean Reciprocal Rank (per-case)
        ================================
        Tìm vị trí 1-indexed đầu tiên mà một ``expected_id`` xuất hiện
        trong ``retrieved_ids``.  MRR = 1 / rank; 0.0 nếu không tìm thấy.

        Parameters
        ----------
        expected_ids  : Ground-Truth doc IDs.
        retrieved_ids : Doc IDs theo thứ tự rank từ agent.
                        Khớp với ``response["retrieved_ids"]`` của MainAgent.

        Returns
        -------
        float : Reciprocal rank ∈ [0.0, 1.0].
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
        expected_set = set(expected_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_set:
                return 1.0 / rank
        return 0.0

    # ------------------------------------------------------------------ #
    #  Batch evaluation (standalone — khi không chạy qua runner)          #
    # ------------------------------------------------------------------ #

    async def evaluate_batch(
        self,
        dataset: List[Dict[str, Any]],
        top_k: int = 3,
        concurrency: int = 10,
        agent_responses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Đánh giá Retrieval trên toàn bộ dataset bất đồng bộ.

        Hỗ trợ hai chế độ:
        ------------------
        **Chế độ 1 – dataset tĩnh** (agent_responses=None):
          Mỗi case trong ``dataset`` cần có:
          - ``expected_retrieval_ids`` : List[str] — Ground-Truth doc IDs.
          - ``retrieved_ids``          : List[str] — doc IDs đã lưu sẵn.

        **Chế độ 2 – kết hợp agent response thật** (agent_responses!=None):
          ``retrieved_ids`` lấy từ ``agent_responses[i]["retrieved_ids"]``
          (là output của MainAgent.query()). ``expected_retrieval_ids`` vẫn
          lấy từ dataset. Cần truyền danh sách response đã collect trước.

        Parameters
        ----------
        dataset         : Danh sách test-case dict.
        top_k           : k dùng cho Hit Rate (mặc định 3).
        concurrency     : Số case song song tối đa (Semaphore).
        agent_responses : Optional list response từ agent, cùng thứ tự dataset.

        Returns
        -------
        Dict chứa:
        - ``per_case``     : List kết quả từng case.
        - ``avg_hit_rate`` : Trung bình Hit Rate @ k (chỉ case ``evaluated``).
        - ``avg_mrr``      : Trung bình MRR (chỉ case ``evaluated``).
        - ``total``        : Tổng case.
        - ``evaluated``    : Số case tính được metric (có cả 2 trường).
        - ``skipped``      : Số case thiếu ít nhất 1 trường bắt buộc.
        - ``no_gt``        : Số case có retrieved_ids nhưng thiếu expected_ids.
        - ``top_k``        : Giá trị k đã dùng.
        """
        if agent_responses is not None and len(agent_responses) != len(dataset):
            raise ValueError(
                f"agent_responses ({len(agent_responses)}) phải cùng độ dài "
                f"với dataset ({len(dataset)})."
            )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._eval_single(
                case=case,
                idx=idx,
                top_k=top_k,
                semaphore=semaphore,
                agent_response=agent_responses[idx] if agent_responses else None,
            )
            for idx, case in enumerate(dataset)
        ]
        per_case: List[Dict[str, Any]] = list(await asyncio.gather(*tasks))

        # Phân loại
        evaluated = [c for c in per_case if c["status"] == "evaluated"]
        skipped = [c for c in per_case if c["status"] == "skipped"]
        no_gt = [c for c in per_case if c["status"] == "no_gt"]

        if evaluated:
            avg_hit_rate = sum(c["hit_rate"] for c in evaluated) / len(evaluated)
            avg_mrr = sum(c["mrr"] for c in evaluated) / len(evaluated)
        else:
            avg_hit_rate = 0.0
            avg_mrr = 0.0

        hit_count = sum(1 for c in evaluated if c["hit_rate"] == 1.0)

        return {
            "per_case": per_case,
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total": len(per_case),
            "evaluated": len(evaluated),
            "skipped": len(skipped),
            "no_gt": len(no_gt),
            "hit_count": hit_count,
            "top_k": top_k,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    async def _eval_single(
        self,
        case: Dict[str, Any],
        idx: int,
        top_k: int,
        semaphore: asyncio.Semaphore,
        agent_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Đánh giá một case riêng lẻ.

        Ưu tiên lấy ``retrieved_ids``:
        1. Từ ``agent_response["retrieved_ids"]`` nếu được truyền vào.
        2. Từ ``case["retrieved_ids"]`` nếu đã lưu sẵn trong dataset.
        """
        async with semaphore:
            question: str = case.get("question", f"case_{idx}")
            expected_ids: Optional[List[str]] = case.get("expected_retrieval_ids")

            # Lấy retrieved_ids theo ưu tiên: agent response > dataset field
            if agent_response is not None:
                retrieved_ids: Optional[List[str]] = agent_response.get("retrieved_ids")
            else:
                retrieved_ids = case.get("retrieved_ids")

            # Case: có retrieved nhưng thiếu ground truth → không thể đánh giá
            if retrieved_ids is not None and not expected_ids:
                logger.debug(
                    "Case [%d] '%s': có retrieved_ids nhưng không có "
                    "expected_retrieval_ids — không thể tính metric.",
                    idx, question,
                )
                return {
                    "idx": idx,
                    "question": question,
                    "status": "no_gt",
                    "hit_rate": None,
                    "mrr": None,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "top_k": top_k,
                }

            # Case: thiếu một trong hai → skip
            if expected_ids is None or retrieved_ids is None:
                logger.warning(
                    "Case [%d] '%s' bị skip: thiếu '%s'.",
                    idx,
                    question,
                    "expected_retrieval_ids" if expected_ids is None else "retrieved_ids",
                )
                return {
                    "idx": idx,
                    "question": question,
                    "status": "skipped",
                    "hit_rate": None,
                    "mrr": None,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "top_k": top_k,
                }

            try:
                hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k)
                mrr = self.calculate_mrr(expected_ids, retrieved_ids)
                return {
                    "idx": idx,
                    "question": question,
                    "status": "evaluated",
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "top_k": top_k,
                }
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Case [%d] '%s' lỗi khi tính metrics: %s",
                    idx, question, exc, exc_info=True,
                )
                return {
                    "idx": idx,
                    "question": question,
                    "status": "error",
                    "error": str(exc),
                    "hit_rate": None,
                    "mrr": None,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "top_k": top_k,
                }
