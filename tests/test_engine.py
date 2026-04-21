"""
tests/test_engine.py
====================
Test doc lap cho:
  - engine/retrieval_eval.py  (RetrievalEvaluator)
  - engine/runner.py          (BenchmarkRunner)

Chay:
    python tests/test_engine.py -v

Khong dung main.py hay check_lab.py.
"""
# Force UTF-8 output on Windows
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# ── Thêm project root vào sys.path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner

logging.basicConfig(
    level=logging.WARNING,        # tắt INFO noise trong test
    format="%(levelname)s  %(name)s — %(message)s",
)

# ══════════════════════════════════════════════════════════════════════ #
#  Helpers                                                               #
# ══════════════════════════════════════════════════════════════════════ #

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
_results: List[Dict[str, Any]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  {status}  {name}" + (f"\n         → {detail}" if detail else ""))
    _results.append({"name": name, "passed": condition})


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def load_golden(path: str = "data/golden_set.jsonl", n: int | None = None) -> List[Dict]:
    p = ROOT / path
    if not p.exists():
        print(f"  ⚠️  {path} not found — skipping real-data tests")
        return []
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[:n] if n else rows


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 1 — RetrievalEvaluator: unit tests                             #
# ══════════════════════════════════════════════════════════════════════ #

async def suite_retrieval_unit() -> None:
    section("SUITE 1 · RetrievalEvaluator — unit tests")
    ev = RetrievalEvaluator()

    # hit_rate
    check("hit_rate: expected in top-1",
          ev.calculate_hit_rate(["doc1"], ["doc1", "doc2", "doc3"], 3) == 1.0)
    check("hit_rate: expected in top-3 (rank 3)",
          ev.calculate_hit_rate(["doc3"], ["doc1", "doc2", "doc3"], 3) == 1.0)
    check("hit_rate: expected NOT in top-3",
          ev.calculate_hit_rate(["doc4"], ["doc1", "doc2", "doc3"], 3) == 0.0)
    check("hit_rate: top_k=1, expected at rank 2 → miss",
          ev.calculate_hit_rate(["doc2"], ["doc1", "doc2", "doc3"], 1) == 0.0)
    check("hit_rate: multi expected_ids, one hits",
          ev.calculate_hit_rate(["doc9", "doc2"], ["doc1", "doc2", "doc3"], 3) == 1.0)
    check("hit_rate: empty expected_ids → 0.0",
          ev.calculate_hit_rate([], ["doc1"], 3) == 0.0)
    check("hit_rate: empty retrieved_ids → 0.0",
          ev.calculate_hit_rate(["doc1"], [], 3) == 0.0)

    # mrr
    check("mrr: rank 1 → 1.0",
          ev.calculate_mrr(["doc1"], ["doc1", "doc2", "doc3"]) == 1.0)
    check("mrr: rank 2 → 0.5",
          ev.calculate_mrr(["doc2"], ["doc1", "doc2", "doc3"]) == 0.5)
    check("mrr: rank 3 → 0.333",
          abs(ev.calculate_mrr(["doc3"], ["doc1", "doc2", "doc3"]) - 1/3) < 1e-6)
    check("mrr: not found → 0.0",
          ev.calculate_mrr(["doc9"], ["doc1", "doc2", "doc3"]) == 0.0)
    check("mrr: picks FIRST matching expected_id (multi)",
          ev.calculate_mrr(["doc9", "doc2"], ["doc1", "doc2", "doc3"]) == 0.5)
    check("mrr: empty expected → 0.0",
          ev.calculate_mrr([], ["doc1"]) == 0.0)
    check("mrr: empty retrieved → 0.0",
          ev.calculate_mrr(["doc1"], []) == 0.0)


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 2 — evaluate_batch: static mode                                #
# ══════════════════════════════════════════════════════════════════════ #

async def suite_evaluate_batch_static() -> None:
    section("SUITE 2 · evaluate_batch — static dataset mode")
    ev = RetrievalEvaluator()

    dataset = [
        # evaluated (hit)
        {"question": "Q_hit1",  "expected_retrieval_ids": ["doc1"],       "retrieved_ids": ["doc1", "doc2", "doc3"]},
        # evaluated (hit rank 2)
        {"question": "Q_hit2",  "expected_retrieval_ids": ["doc2"],       "retrieved_ids": ["doc1", "doc2", "doc3"]},
        # evaluated (miss)
        {"question": "Q_miss",  "expected_retrieval_ids": ["doc9"],       "retrieved_ids": ["doc1", "doc2", "doc3"]},
        # evaluated: empty expected_ids (out-of-scope case) → both 0.0
        {"question": "Q_oos",   "expected_retrieval_ids": [],             "retrieved_ids": ["doc1"]},
        # skipped: missing retrieved_ids
        {"question": "Q_skip1", "expected_retrieval_ids": ["doc1"]},
        # no_gt: has retrieved, no expected
        {"question": "Q_nogt",  "retrieved_ids": ["doc1"]},
        # skipped: missing both
        {"question": "Q_skip2"},
    ]

    r = await ev.evaluate_batch(dataset, top_k=3)

    check("total == 7",            r["total"] == 7,    str(r["total"]))
    check("evaluated == 4",        r["evaluated"] == 4, str(r["evaluated"]))
    check("skipped == 2",          r["skipped"] == 2,  str(r["skipped"]))
    check("no_gt == 1",            r["no_gt"] == 1,    str(r["no_gt"]))

    # avg_hit_rate: Q_hit1=1, Q_hit2=1, Q_miss=0, Q_oos=0  → 2/4 = 0.5
    check("avg_hit_rate == 0.5",
          abs(r["avg_hit_rate"] - 0.5) < 1e-4, str(r["avg_hit_rate"]))
    # avg_mrr: 1 + 0.5 + 0 + 0  → 1.5/4 = 0.375
    check("avg_mrr == 0.375",
          abs(r["avg_mrr"] - 0.375) < 1e-4, str(r["avg_mrr"]))
    check("hit_count == 2",        r["hit_count"] == 2, str(r["hit_count"]))
    check("top_k stored == 3",     r["top_k"] == 3)

    # per_case statuses
    statuses = [c["status"] for c in r["per_case"]]
    check("per_case statuses correct",
          statuses == ["evaluated","evaluated","evaluated","evaluated","skipped","no_gt","skipped"],
          str(statuses))


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 3 — evaluate_batch: agent_response mode                        #
# ══════════════════════════════════════════════════════════════════════ #

async def suite_evaluate_batch_agent_mode() -> None:
    section("SUITE 3 · evaluate_batch — agent_response mode")
    ev = RetrievalEvaluator()

    dataset = [
        {"question": "Q1", "expected_retrieval_ids": ["doc1"]},
        {"question": "Q2", "expected_retrieval_ids": ["doc2"]},
        {"question": "Q3", "expected_retrieval_ids": ["doc9"]},   # miss
        {"question": "Q4"},                                         # no_gt
    ]
    agent_responses = [
        {"retrieved_ids": ["doc1", "doc2"]},
        {"retrieved_ids": ["doc3", "doc2"]},
        {"retrieved_ids": ["doc1", "doc2"]},
        {"retrieved_ids": ["doc1"]},          # no expected → no_gt
    ]

    r = await ev.evaluate_batch(dataset, top_k=3, agent_responses=agent_responses)
    check("evaluated == 3",    r["evaluated"] == 3,  str(r["evaluated"]))
    check("no_gt == 1",        r["no_gt"] == 1,      str(r["no_gt"]))
    check("avg_hit_rate == 0.6667",
          abs(r["avg_hit_rate"] - 2/3) < 1e-4, str(r["avg_hit_rate"]))
    check("avg_mrr == 0.5",
          abs(r["avg_mrr"] - (1+0.5+0)/3) < 1e-4, str(r["avg_mrr"]))

    # length mismatch should raise
    try:
        await ev.evaluate_batch(dataset, agent_responses=agent_responses[:2])
        check("ValueError on length mismatch", False, "no error raised")
    except ValueError:
        check("ValueError on length mismatch", True)


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 4 — evaluate_batch: with real golden_set.jsonl (static)        #
# ══════════════════════════════════════════════════════════════════════ #

async def suite_real_data_static() -> None:
    section("SUITE 4 · evaluate_batch — real golden_set.jsonl (static retrieved_ids absent)")
    ev = RetrievalEvaluator()
    dataset = load_golden()
    if not dataset:
        return

    # golden_set không có retrieved_ids tĩnh → tất cả phải skipped hoặc no_gt
    r = await ev.evaluate_batch(dataset, top_k=3)
    check(f"total == {len(dataset)}", r["total"] == len(dataset))
    check("evaluated == 0 (no retrieved_ids in dataset)",
          r["evaluated"] == 0, str(r["evaluated"]))
    check("skipped + no_gt == total",
          r["skipped"] + r["no_gt"] == r["total"],
          f"skipped={r['skipped']} no_gt={r['no_gt']}")


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 5 — BenchmarkRunner: Fake components, error isolation          #
# ══════════════════════════════════════════════════════════════════════ #

class _FakeAgent:
    """Agent giả: trả retrieved_ids thật, simulate crash cho 1 case."""
    async def query(self, question: str) -> Dict[str, Any]:
        if "crash" in question.lower():
            raise RuntimeError("Simulated agent crash")
        await asyncio.sleep(0.01)
        # Trả đúng schema của MainAgent
        return {
            "answer": f"Answer for: {question}",
            "retrieved_ids": ["doc_it_helpdesk_faq", "doc_sla_p1_2026"],
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens_used": 80,
                "sources": ["doc_it_helpdesk_faq"],
                "llm_mode": "fallback_no_api_key",
            },
        }

class _FakeEval:
    async def score(self, case: Dict, resp: Dict) -> Dict[str, Any]:
        return {"faithfulness": 0.9, "relevancy": 0.85}

class _FakeJudge:
    """Judge giả: trả đúng schema của MultiModelJudge."""
    async def evaluate_multi_judge(self, q: str, a: str, gt: str) -> Dict[str, Any]:
        return {
            "final_score": 4.0,
            "agreement_rate": 0.9,
            "agreement_observed_criteria": 3,
            "conflict_detected": False,
            "judge_errors": 0,
            "reasoning": "Good answer",
            "scoring_mode": "heuristic",
            "individual_scores": {
                "gpt-4o": {"accuracy_score": 4, "faithfulness_score": 4, "relevancy_score": 4},
                "gpt-4o-mini": {"accuracy_score": 4, "faithfulness_score": 4, "relevancy_score": 4},
            },
        }

async def suite_runner_fake() -> None:
    section("SUITE 5 · BenchmarkRunner — Fake components (error isolation)")

    dataset = [
        {"question": "Tôi quên mật khẩu?",    "expected_answer": "...", "expected_retrieval_ids": ["doc_it_helpdesk_faq"]},
        {"question": "SLA P1 là bao lâu?",     "expected_answer": "...", "expected_retrieval_ids": ["doc_sla_p1_2026"]},
        {"question": "CRASH this case",         "expected_answer": "..."},   # sẽ error
        {"question": "Chính sách nghỉ phép?",  "expected_answer": "...", "expected_retrieval_ids": ["doc_hr_leave_policy"]},
        {"question": "Hoàn tiền Flash Sale?",  "expected_answer": "...", "expected_retrieval_ids": ["doc_policy_refund_v4"]},
    ]

    runner = BenchmarkRunner(_FakeAgent(), _FakeEval(), _FakeJudge(),
                             concurrency=3, top_k=3)
    t0 = time.perf_counter()
    results = await runner.run_all(dataset, batch_size=3)
    elapsed = time.perf_counter() - t0

    check("run_all returns 5 results",   len(results) == 5, str(len(results)))
    statuses = [r["status"] for r in results]
    check("crash case → error",          results[2]["status"] == "error", str(statuses))
    check("non-crash cases → pass",      statuses.count("pass") == 4, str(statuses))
    check("error case has 'error' key",  "error" in results[2] and results[2]["error"])

    # Retrieval inline
    r0 = results[0]
    check("retrieval field present",     r0.get("retrieval") is not None)
    check("hit_rate for Q0 == 1.0 (doc_it_helpdesk_faq retrieved)",
          r0["retrieval"]["hit_rate"] == 1.0, str(r0["retrieval"]["hit_rate"]))
    check("mrr for Q0 == 1.0 (rank 1)",
          r0["retrieval"]["mrr"] == 1.0, str(r0["retrieval"]["mrr"]))

    r1 = results[1]
    check("hit_rate for Q1 == 1.0 (doc_sla_p1_2026 at rank 2)",
          r1["retrieval"]["hit_rate"] == 1.0, str(r1["retrieval"]["hit_rate"]))
    check("mrr for Q1 == 0.5 (rank 2)",
          r1["retrieval"]["mrr"] == 0.5, str(r1["retrieval"]["mrr"]))

    r3 = results[3]
    check("hit_rate Q3 miss (doc_hr_leave_policy not retrieved)",
          r3["retrieval"]["hit_rate"] == 0.0, str(r3["retrieval"]["hit_rate"]))

    # Judge field
    check("judge.final_score == 4.0",    r0["judge"]["final_score"] == 4.0)
    check("judge.agreement_rate == 0.9", r0["judge"]["agreement_rate"] == 0.9)
    check("judge.individual_scores present",
          isinstance(r0["judge"].get("individual_scores"), dict))
    check("judge.conflict_detected == False",
          r0["judge"]["conflict_detected"] == False)
    check("judge.scoring_mode == heuristic",
          r0["judge"]["scoring_mode"] == "heuristic")

    # Cost field
    check("cost.tokens_used == 80",      r0["cost"]["tokens_used"] == 80)
    check("cost.llm_mode == fallback",   r0["cost"]["llm_mode"] == "fallback_no_api_key")
    check("cost.model present",          r0["cost"]["model"] == "gpt-4o-mini")

    # Performance
    check(f"ran in <5s (actual {elapsed:.2f}s)", elapsed < 5.0, f"{elapsed:.2f}s")

    # aggregate_results
    agg = BenchmarkRunner.aggregate_results(results)
    check("agg.counts.total == 5",       agg["counts"]["total"] == 5)
    check("agg.counts.pass == 4",        agg["counts"]["pass"] == 4)
    check("agg.counts.error == 1",       agg["counts"]["error"] == 1)
    check("agg.retrieval.avg_hit_rate present",
          isinstance(agg["retrieval"]["avg_hit_rate"], float))
    check("agg.cost.total_tokens == 320", agg["cost"]["total_tokens"] == 320,
          str(agg["cost"]["total_tokens"]))
    check("agg.judge.avg_final_score == 4.0",
          agg["judge"]["avg_final_score"] == 4.0, str(agg["judge"]["avg_final_score"]))


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 6 — BenchmarkRunner: real MainAgent (no API key, fallback)     #
# ══════════════════════════════════════════════════════════════════════ #

async def suite_runner_real_agent() -> None:
    section("SUITE 6 · BenchmarkRunner — Real MainAgent + MultiModelJudge (heuristic)")
    dataset = load_golden(n=5)   # chỉ 5 case để test nhanh
    if not dataset:
        return

    try:
        from agent.main_agent import MainAgent
        from engine.llm_judge import MultiModelJudge
    except ImportError as e:
        print(f"  ⚠️  Import error: {e} — skip suite 6")
        return

    class _SimpleEval:
        async def score(self, case, resp):
            return {"faithfulness": 0.8, "relevancy": 0.8}

    try:
        agent  = MainAgent(docs_dir=str(ROOT / "docs"))
        judge  = MultiModelJudge()
        runner = BenchmarkRunner(agent, _SimpleEval(), judge, concurrency=3, top_k=3)

        t0 = time.perf_counter()
        results = await runner.run_all(dataset, batch_size=5)
        elapsed = time.perf_counter() - t0

        check("all 5 cases returned",    len(results) == 5)
        check("no errors (fallback OK)", all(r["status"] != "error" for r in results),
              str([r["status"] for r in results]))
        check("retrieved_ids non-empty for most cases",
              sum(1 for r in results
                  if r.get("retrieval") and r["retrieval"]["retrieved_ids"]) >= 3)
        check("scoring_mode == heuristic (no API key)",
              all(r["judge"]["scoring_mode"] == "heuristic"
                  for r in results if r.get("judge")))

        # Retrieval vs golden expected
        gt_cases = [r for r in results
                    if r.get("retrieval") and r["retrieval"]["has_ground_truth"]]
        if gt_cases:
            hit_rate = sum(r["retrieval"]["hit_rate"] for r in gt_cases) / len(gt_cases)
            mrr_avg  = sum(r["retrieval"]["mrr"]      for r in gt_cases) / len(gt_cases)
            print(f"\n  📊 Real retrieval metrics (n={len(gt_cases)}):")
            print(f"     avg_hit_rate@3 = {hit_rate:.4f}")
            print(f"     avg_mrr        = {mrr_avg:.4f}")
            print(f"     elapsed        = {elapsed:.2f}s")
            check("hit_rate@3 > 0 (retriever finds something)",
                  hit_rate > 0, f"{hit_rate:.4f}")

    except Exception:
        print(f"  ⚠️  Suite 6 failed:\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════ #
#  SUITE 7 — BenchmarkRunner: progress_callback + concurrency           #
# ══════════════════════════════════════════════════════════════════════ #

async def suite_runner_progress_callback() -> None:
    section("SUITE 7 · BenchmarkRunner — progress_callback + concurrency ordering")

    calls: List[int] = []
    def _cb(done: int, total: int, result: Dict) -> None:
        calls.append(done)

    dataset = [
        {"question": f"q{i}", "expected_answer": "a", "expected_retrieval_ids": ["doc1"]}
        for i in range(8)
    ]

    runner = BenchmarkRunner(_FakeAgent(), _FakeEval(), _FakeJudge(),
                             concurrency=4, top_k=3)
    results = await runner.run_all(dataset, batch_size=4, progress_callback=_cb)

    check("8 results returned",        len(results) == 8)
    check("callback called 8 times",   len(calls) == 8, str(len(calls)))
    check("callback done values 1–8",  sorted(calls) == list(range(1, 9)), str(calls))
    check("result order preserved",
          [r["idx"] for r in results] == list(range(8)),
          str([r["idx"] for r in results]))


# ══════════════════════════════════════════════════════════════════════ #
#  Runner                                                                #
# ══════════════════════════════════════════════════════════════════════ #

async def main() -> None:
    print("\n" + "═"*60)
    print("  ENGINE TEST SUITE")
    print("  retrieval_eval.py + runner.py")
    print("═"*60)

    await suite_retrieval_unit()
    await suite_evaluate_batch_static()
    await suite_evaluate_batch_agent_mode()
    await suite_real_data_static()
    await suite_runner_fake()
    await suite_runner_real_agent()
    await suite_runner_progress_callback()

    # ── Summary ──────────────────────────────────────────────────────
    total   = len(_results)
    passed  = sum(1 for r in _results if r["passed"])
    failed  = total - passed

    print("\n" + "═"*60)
    print(f"  RESULT: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED:")
        for r in _results:
            if not r["passed"]:
                print(f"    ✗ {r['name']}")
    else:
        print("  — all green 🎉")
    print("═"*60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
