import asyncio
import json
import os
import time
from typing import Any, Dict, List, Tuple

from agent.main_agent import MainAgent
from engine.llm_judge import MultiModelJudge
from engine.runner import BenchmarkRunner

MIN_AVG_SCORE = 3.0
MIN_HIT_RATE = 0.7
MIN_AGREEMENT_RATE = 0.6
MAX_ERROR_RATE = 0.05
MIN_DELTA_SCORE = -0.05
MIN_DELTA_HIT_RATE = -0.05


class ExpertEvaluator:
    async def score(self, case, resp):
        # Placeholder custom metrics (retrieval được tính inline trong runner)
        return {
            "faithfulness": 0.9,
            "relevancy": 0.8,
        }


def _load_dataset(path: str = "data/golden_set.jsonl") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_runner(agent_model: str) -> BenchmarkRunner:
    return BenchmarkRunner(
        agent=MainAgent(model=agent_model),
        evaluator=ExpertEvaluator(),
        judge=MultiModelJudge(),
    )


def _build_gate(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    v1_metrics = v1["metrics"]
    v2_metrics = v2["metrics"]
    delta_score = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    delta_hit = v2_metrics["hit_rate"] - v1_metrics["hit_rate"]

    rules = {
        "avg_score": v2_metrics["avg_score"] >= MIN_AVG_SCORE,
        "hit_rate": v2_metrics["hit_rate"] >= MIN_HIT_RATE,
        "agreement_rate": v2_metrics["agreement_rate"] >= MIN_AGREEMENT_RATE,
        "error_rate": v2_metrics.get("error_rate", 0.0) <= MAX_ERROR_RATE,
        "delta_score": delta_score >= MIN_DELTA_SCORE,
        "delta_hit_rate": delta_hit >= MIN_DELTA_HIT_RATE,
    }
    failed_rules = [name for name, passed in rules.items() if not passed]

    return {
        "approve": len(failed_rules) == 0,
        "failed_rules": failed_rules,
        "thresholds": {
            "min_avg_score": MIN_AVG_SCORE,
            "min_hit_rate": MIN_HIT_RATE,
            "min_agreement_rate": MIN_AGREEMENT_RATE,
            "max_error_rate": MAX_ERROR_RATE,
            "min_delta_score": MIN_DELTA_SCORE,
            "min_delta_hit_rate": MIN_DELTA_HIT_RATE,
        },
        "deltas": {
            "avg_score": round(delta_score, 4),
            "hit_rate": round(delta_hit, 4),
        },
    }


async def run_benchmark_with_results(
    agent_version: str,
    agent_model: str,
    dataset_path: str = "data/golden_set.jsonl",
) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Any] | None]:
    print(f"🚀 Khởi động Benchmark cho {agent_version} (model={agent_model})...")

    if not os.path.exists(dataset_path):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    dataset = _load_dataset(dataset_path)
    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    batch_size = int(os.getenv("BENCHMARK_BATCH_SIZE", "5"))
    runner = _build_runner(agent_model)
    results = await runner.run_all(dataset, batch_size=batch_size)
    agg = runner.aggregate_results(results)

    total = agg["counts"]["total"]
    error_rate = (agg["counts"]["error"] / total) if total else 0.0

    summary = {
        "metadata": {
            "version": agent_version,
            "agent_model": agent_model,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": batch_size,
        },
        "metrics": {
            "avg_score": agg["judge"]["avg_final_score"],
            "hit_rate": agg["retrieval"]["avg_hit_rate"],
            "mrr": agg["retrieval"]["avg_mrr"],
            "agreement_rate": agg["judge"]["avg_agreement_rate"],
            "conflict_rate": agg["judge"]["conflict_rate"],
            "error_rate": round(error_rate, 4),
        },
        "counts": agg["counts"],
        "cost": agg["cost"],
    }
    return results, summary


async def run_benchmark(version: str, model: str):
    _, summary = await run_benchmark_with_results(version, model)
    return summary


async def main():
    v1_model = os.getenv("AGENT_V1_MODEL", "gpt-4.1-nano")
    v2_model = os.getenv("AGENT_V2_MODEL", "gpt-4o-mini")

    v1_summary = await run_benchmark("Agent_V1_Base", v1_model)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", v2_model)

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    gate = _build_gate(v1_summary, v2_summary)

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(
        f"V2 Hit Rate: {v2_summary['metrics']['hit_rate']:.4f} | "
        f"Agreement: {v2_summary['metrics']['agreement_rate']:.4f} | "
        f"Error Rate: {v2_summary['metrics']['error_rate']:.4f}"
    )

    os.makedirs("reports", exist_ok=True)
    summary_output = {
        **v2_summary,
        "regression": {
            "v1": v1_summary["metrics"],
            "v2": v2_summary["metrics"],
            "deltas": gate["deltas"],
        },
        "release_gate": gate,
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if gate["approve"]:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        if gate["failed_rules"]:
            print(f"⚠️ Rule fail: {', '.join(gate['failed_rules'])}")
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
