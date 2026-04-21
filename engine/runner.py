"""
engine/runner.py
================
Async BenchmarkRunner khớp với interface thực tế của:

- ``agent/main_agent.py``  (MainAgent)
  · query(question) → {answer, retrieved_ids, metadata{model, tokens_used, sources, llm_mode}}

- ``engine/llm_judge.py``  (MultiModelJudge)
  · evaluate_multi_judge(q, a, gt) → {
        individual_scores, final_score, agreement_rate,
        agreement_observed_criteria, conflict_detected,
        judge_errors, reasoning, scoring_mode
    }

- ``engine/retrieval_eval.py``  (RetrievalEvaluator)
  · calculate_hit_rate(expected_ids, retrieved_ids, top_k) → float
  · calculate_mrr(expected_ids, retrieved_ids) → float
  · evaluate_batch(dataset, top_k, concurrency) → Dict   ← dùng ở ngoài runner

Tính năng:
1. Batch concurrency   – asyncio.Semaphore ngăn rate-limit.
2. Error isolation     – mỗi case bọc try/except riêng; crash 1 case ≠ dừng batch.
3. Retrieval in-line   – hit_rate@k + MRR tính ngay từ retrieved_ids của agent response.
4. Cost tracking       – gom tokens_used + scoring_mode cho từng case.
5. Progress callback   – hook tuỳ chọn (done, total, result) → None.
6. Logging chi tiết    – batch summary + final summary.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from engine.retrieval_eval import RetrievalEvaluator

logger = logging.getLogger(__name__)

# Ngưỡng pass/fail: final_score là avg của 3 criteria trên thang 1-5
# → 3.0 tương đương "acceptable" (60%)
_PASS_THRESHOLD: float = 3.0


class BenchmarkRunner:
    """
    Chạy pipeline đánh giá đầy đủ trên toàn bộ dataset:

    Agent query → Retrieval eval (hit_rate, MRR) → RAGAS/custom eval
    → Multi-Judge (primary + secondary ± tiebreaker) → aggregate result.
    """

    def __init__(
        self,
        agent: Any,
        evaluator: Any,
        judge: Any,
        concurrency: int = 5,
        top_k: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        agent       : MainAgent — async query(question) → Dict.
        evaluator   : ExpertEvaluator (hoặc tương tự) — async score(case, resp) → Dict.
        judge       : MultiModelJudge — async evaluate_multi_judge(q, a, gt) → Dict.
        concurrency : Số case LLM được phép chạy đồng thời (Semaphore).
        top_k       : k dùng cho hit_rate@k khi tính retrieval inline.
        """
        if concurrency <= 0:
            raise ValueError(f"concurrency phải > 0, nhận {concurrency}.")
        if top_k <= 0:
            raise ValueError(f"top_k phải > 0, nhận {top_k}.")
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.concurrency = concurrency
        self.top_k = top_k
        self._retrieval_evaluator = RetrievalEvaluator()

    @staticmethod
    def _as_id_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            normalized = [str(item).strip() for item in value if item is not None]
            return [item for item in normalized if item]
        return []

    # ------------------------------------------------------------------ #
    #  Single-case execution                                               #
    # ------------------------------------------------------------------ #

    async def run_single_test(
        self,
        test_case: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        case_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Chạy pipeline đánh giá cho một test-case, có error isolation.

        Luồng xử lý
        ------------
        1. Agent.query()  → answer + retrieved_ids + metadata.
        2. Retrieval eval → hit_rate@k, MRR (dùng retrieved_ids từ agent + 
                            expected_retrieval_ids từ dataset).
        3. Evaluator.score() → ragas / custom metrics.
        4. Judge.evaluate_multi_judge() → final_score, agreement_rate, ...
        5. Gom kết quả, tính pass/fail/error.

        Mọi exception trong pipeline → status="error", không re-raise,
        không ảnh hưởng các case khác trong cùng batch.
        """
        question: str = test_case.get("question", f"<case_{case_idx}>")

        async with semaphore:
            start_time = time.perf_counter()
            try:
                # ── Step 1: Agent query ──────────────────────────────── #
                response: Dict[str, Any] = await self.agent.query(question)
                latency = time.perf_counter() - start_time

                answer: str = response.get("answer", "")
                retrieved_ids: List[str] = self._as_id_list(response.get("retrieved_ids"))
                agent_meta: Dict[str, Any] = response.get("metadata", {})

                # ── Step 2: Retrieval eval (inline, no extra I/O) ─────── #
                expected_ids: List[str] = self._as_id_list(
                    test_case.get("expected_retrieval_ids")
                )
                hit_rate = self._retrieval_evaluator.calculate_hit_rate(
                    expected_ids, retrieved_ids, self.top_k
                )
                mrr = self._retrieval_evaluator.calculate_mrr(
                    expected_ids, retrieved_ids
                )
                retrieval_result: Dict[str, Any] = {
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "top_k": self.top_k,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    # Đánh dấu nếu không có expected_ids để distinguish "miss" vs "unknown"
                    "has_ground_truth": bool(expected_ids),
                }

                # ── Step 3: RAGAS / custom evaluator ─────────────────── #
                ragas_scores: Dict[str, Any] = await self.evaluator.score(
                    test_case, response
                )

                # ── Step 4: Multi-Judge ───────────────────────────────── #
                ground_truth: str = test_case.get("expected_answer", "")
                judge_result: Dict[str, Any] = await self.judge.evaluate_multi_judge(
                    question, answer, ground_truth
                )

                # ── Step 5: Assemble ──────────────────────────────────── #
                final_score: float = judge_result.get("final_score", 0.0)

                # Cost / token tracking từ agent metadata
                tokens_used: int = agent_meta.get("tokens_used", 0)

                return {
                    "idx": case_idx,
                    "question": question,
                    "agent_response": answer,
                    "latency_sec": round(latency, 4),
                    # Retrieval metrics (inline)
                    "retrieval": retrieval_result,
                    # RAGAS / custom
                    "ragas": ragas_scores,
                    # Multi-judge full breakdown
                    "judge": {
                        "final_score": judge_result.get("final_score"),
                        "agreement_rate": judge_result.get("agreement_rate"),
                        "agreement_observed_criteria": judge_result.get(
                            "agreement_observed_criteria"
                        ),
                        "conflict_detected": judge_result.get("conflict_detected"),
                        "judge_errors": judge_result.get("judge_errors"),
                        "reasoning": judge_result.get("reasoning"),
                        "scoring_mode": judge_result.get("scoring_mode"),
                        "individual_scores": judge_result.get("individual_scores"),
                    },
                    # Cost tracking
                    "cost": {
                        "tokens_used": tokens_used,
                        "llm_mode": agent_meta.get("llm_mode"),
                        "model": agent_meta.get("model"),
                        "sources": agent_meta.get("sources", []),
                    },
                    "status": "pass" if final_score >= _PASS_THRESHOLD else "fail",
                }

            except Exception:  # pylint: disable=broad-except
                latency = time.perf_counter() - start_time
                tb = traceback.format_exc()
                logger.error(
                    "❌  Case [%d] '%s' failed after %.2fs:\n%s",
                    case_idx,
                    question,
                    latency,
                    tb,
                )
                return {
                    "idx": case_idx,
                    "question": question,
                    "agent_response": None,
                    "latency_sec": round(latency, 4),
                    "retrieval": None,
                    "ragas": None,
                    "judge": None,
                    "cost": None,
                    "status": "error",
                    "error": tb.strip().splitlines()[-1],
                    "traceback": tb,
                }

    # ------------------------------------------------------------------ #
    #  Batch runner                                                        #
    # ------------------------------------------------------------------ #

    async def run_all(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 5,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chạy toàn bộ dataset với kiểm soát concurrency và log tiến trình.

        Strategy
        --------
        - Chia dataset thành nhóm ``batch_size`` case để log tiến trình.
        - Tất cả case trong batch được schedule đồng thời qua asyncio.gather,
          nhưng bị giới hạn thực thi bởi Semaphore(self.concurrency).
        - ``return_exceptions=True`` làm lưới an toàn thứ 2 ngoài try/except.

        Parameters
        ----------
        dataset           : Danh sách test-case dict.
        batch_size        : Số case mỗi nhóm log.
        progress_callback : (done: int, total: int, result: Dict) → None.

        Returns
        -------
        List[Dict] kết quả theo đúng thứ tự input.
        """
        if not dataset:
            logger.warning("Dataset rỗng — không có case nào để chạy.")
            return []
        if batch_size <= 0:
            raise ValueError(f"batch_size phải > 0, nhận {batch_size}.")

        total = len(dataset)
        semaphore = asyncio.Semaphore(self.concurrency)
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "🚀 Bắt đầu benchmark: %d cases | batch_size=%d | concurrency=%d | top_k=%d",
            total,
            batch_size,
            self.concurrency,
            self.top_k,
        )

        all_results: List[Dict[str, Any]] = []
        wall_start = time.perf_counter()

        for batch_start in range(0, total, batch_size):
            batch = dataset[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1

            logger.info(
                "  📦 Batch %d/%d — cases %d–%d",
                batch_num,
                total_batches,
                batch_start,
                batch_start + len(batch) - 1,
            )

            tasks = [
                self.run_single_test(case, semaphore, batch_start + i)
                for i, case in enumerate(batch)
            ]

            # return_exceptions=True: nếu run_single_test vẫn leak exception
            # (không nên xảy ra), gather không crash toàn batch
            batch_results: List[Any] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # Xử lý exception rò (safety net)
            for i, res in enumerate(batch_results):
                if isinstance(res, BaseException):
                    case_idx = batch_start + i
                    question = batch[i].get("question", f"<case_{case_idx}>")
                    logger.error(
                        "⚠️  Unexpected exception leaked from run_single_test "
                        "case [%d] '%s': %s",
                        case_idx,
                        question,
                        res,
                    )
                    res = {
                        "idx": case_idx,
                        "question": question,
                        "agent_response": None,
                        "latency_sec": 0.0,
                        "retrieval": None,
                        "ragas": None,
                        "judge": None,
                        "cost": None,
                        "status": "error",
                        "error": str(res),
                        "traceback": "",
                    }
                    batch_results[i] = res

                # Progress callback
                if progress_callback is not None:
                    done = len(all_results) + i + 1
                    try:
                        progress_callback(done, total, res)
                    except Exception:  # pylint: disable=broad-except
                        pass

            all_results.extend(batch_results)
            self._log_batch_summary(batch_results, batch_num, total_batches)

        wall_elapsed = time.perf_counter() - wall_start
        self._log_final_summary(all_results, wall_elapsed)
        return all_results

    # ------------------------------------------------------------------ #
    #  Aggregate helpers (dùng bởi main.py)                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tính các chỉ số tổng hợp từ danh sách kết quả của run_all().

        Trả về dict phù hợp với schema reports/summary.json bao gồm:
        - retrieval: avg_hit_rate, avg_mrr
        - judge: avg_final_score, avg_agreement_rate, conflict_rate
        - cost: total_tokens, avg_tokens_per_case
        - performance: avg_latency_sec, throughput_cps (nếu truyền elapsed)
        """
        valid = [r for r in results if r.get("status") in ("pass", "fail")]
        total = len(results)
        n_valid = len(valid)

        if n_valid == 0:
            return {
                "retrieval": {"avg_hit_rate": 0.0, "avg_mrr": 0.0},
                "judge": {
                    "avg_final_score": 0.0,
                    "avg_agreement_rate": 0.0,
                    "conflict_rate": 0.0,
                },
                "cost": {"total_tokens": 0, "avg_tokens_per_case": 0},
                "counts": {
                    "total": total,
                    "pass": 0,
                    "fail": 0,
                    "error": total,
                    "evaluated": 0,
                },
            }

        # Retrieval (chỉ các case có ground truth)
        retrieval_cases = [
            r for r in valid
            if r.get("retrieval") and r["retrieval"].get("has_ground_truth")
        ]
        avg_hit_rate = (
            sum(r["retrieval"]["hit_rate"] for r in retrieval_cases) / len(retrieval_cases)
            if retrieval_cases else 0.0
        )
        avg_mrr = (
            sum(r["retrieval"]["mrr"] for r in retrieval_cases) / len(retrieval_cases)
            if retrieval_cases else 0.0
        )

        # Judge
        judge_cases = [r for r in valid if r.get("judge") and r["judge"].get("final_score") is not None]
        avg_score = (
            sum(r["judge"]["final_score"] for r in judge_cases) / len(judge_cases)
            if judge_cases else 0.0
        )
        agreement_values = [
            r["judge"]["agreement_rate"]
            for r in judge_cases
            if r["judge"].get("agreement_rate") is not None
        ]
        avg_agreement = (
            sum(agreement_values) / len(agreement_values)
            if agreement_values else 0.0
        )
        conflict_count = sum(
            1 for r in judge_cases if r["judge"].get("conflict_detected")
        )
        conflict_rate = conflict_count / len(judge_cases) if judge_cases else 0.0

        # Cost
        total_tokens = sum(
            r["cost"]["tokens_used"]
            for r in valid
            if r.get("cost") and r["cost"].get("tokens_used")
        )

        # Counts
        n_pass = sum(1 for r in results if r.get("status") == "pass")
        n_fail = sum(1 for r in results if r.get("status") == "fail")
        n_error = sum(1 for r in results if r.get("status") == "error")

        return {
            "retrieval": {
                "avg_hit_rate": round(avg_hit_rate, 4),
                "avg_mrr": round(avg_mrr, 4),
                "evaluated_cases": len(retrieval_cases),
            },
            "judge": {
                "avg_final_score": round(avg_score, 4),
                "avg_agreement_rate": round(avg_agreement, 4),
                "conflict_rate": round(conflict_rate, 4),
            },
            "cost": {
                "total_tokens": total_tokens,
                "avg_tokens_per_case": round(total_tokens / n_valid, 1) if n_valid else 0,
            },
            "counts": {
                "total": total,
                "pass": n_pass,
                "fail": n_fail,
                "error": n_error,
                "evaluated": n_valid,
            },
        }

    # ------------------------------------------------------------------ #
    #  Logging helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_batch_summary(
        results: List[Any],
        batch_num: int,
        total_batches: int,
    ) -> None:
        passed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "pass")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "fail")
        errors = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error")
        avg_lat = (
            sum(r.get("latency_sec", 0) for r in results if isinstance(r, dict))
            / len(results)
            if results else 0
        )
        logger.info(
            "  ✅ Batch %d/%d done — pass=%d fail=%d error=%d | avg_latency=%.2fs",
            batch_num, total_batches, passed, failed, errors, avg_lat,
        )

    @staticmethod
    def _log_final_summary(results: List[Dict[str, Any]], elapsed: float) -> None:
        total = len(results)
        passed = sum(1 for r in results if r.get("status") == "pass")
        failed = sum(1 for r in results if r.get("status") == "fail")
        errors = sum(1 for r in results if r.get("status") == "error")
        avg_lat = (
            sum(r.get("latency_sec", 0) for r in results) / total if total else 0
        )
        total_tokens = sum(
            r["cost"]["tokens_used"]
            for r in results
            if r.get("cost") and r["cost"].get("tokens_used")
        )
        logger.info(
            "\n📊 Benchmark hoàn tất — %d cases trong %.2fs\n"
            "   pass=%d (%.1f%%)  fail=%d (%.1f%%)  error=%d (%.1f%%)\n"
            "   avg_latency=%.2fs  throughput=%.1f cases/s\n"
            "   total_tokens=%d",
            total, elapsed,
            passed, 100 * passed / total if total else 0,
            failed, 100 * failed / total if total else 0,
            errors, 100 * errors / total if total else 0,
            avg_lat,
            total / elapsed if elapsed > 0 else 0,
            total_tokens,
        )
