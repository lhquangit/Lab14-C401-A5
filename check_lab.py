import json
import os
from typing import Any, Dict, List


REQUIRED_FILES = [
    "reports/summary.json",
    "reports/benchmark_results.json",
    "analysis/failure_analysis.md",
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_metric_range(name: str, value: Any, lo: float, hi: float, errors: List[str]) -> None:
    if not _is_number(value):
        errors.append(f"metrics.{name} phải là số.")
        return
    if value < lo or value > hi:
        errors.append(f"metrics.{name}={value} nằm ngoài khoảng [{lo}, {hi}].")


def _validate_summary_schema(summary: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    if "metadata" not in summary or "metrics" not in summary:
        errors.append("summary.json thiếu 'metadata' hoặc 'metrics'.")
        return

    metadata = summary["metadata"]
    metrics = summary["metrics"]

    for field in ("version", "total", "timestamp"):
        if field not in metadata:
            errors.append(f"metadata thiếu trường '{field}'.")

    required_metrics = ["avg_score", "hit_rate", "mrr", "agreement_rate", "conflict_rate"]
    for field in required_metrics:
        if field not in metrics:
            errors.append(f"metrics thiếu trường '{field}'.")

    if errors:
        return

    if not isinstance(metadata["total"], int) or metadata["total"] < 0:
        errors.append("metadata.total phải là số nguyên không âm.")

    _validate_metric_range("avg_score", metrics["avg_score"], 0.0, 5.0, errors)
    _validate_metric_range("hit_rate", metrics["hit_rate"], 0.0, 1.0, errors)
    _validate_metric_range("mrr", metrics["mrr"], 0.0, 1.0, errors)
    _validate_metric_range("agreement_rate", metrics["agreement_rate"], 0.0, 1.0, errors)
    _validate_metric_range("conflict_rate", metrics["conflict_rate"], 0.0, 1.0, errors)

    if "error_rate" in metrics:
        _validate_metric_range("error_rate", metrics["error_rate"], 0.0, 1.0, errors)

    if metadata["total"] < 50:
        warnings.append(
            f"Tổng số case hiện tại là {metadata['total']} (<50). Có thể không đạt tiêu chí rubric."
        )

    counts = summary.get("counts")
    if counts is not None:
        for field in ("total", "pass", "fail", "error", "evaluated"):
            if field not in counts:
                errors.append(f"counts thiếu trường '{field}'.")
            elif not isinstance(counts[field], int) or counts[field] < 0:
                errors.append(f"counts.{field} phải là số nguyên không âm.")

        if not errors and (counts["pass"] + counts["fail"] + counts["error"] != counts["total"]):
            errors.append("counts không nhất quán: pass+fail+error != total.")

    cost = summary.get("cost")
    if cost is not None:
        if "total_tokens" not in cost or "avg_tokens_per_case" not in cost:
            errors.append("cost phải có 'total_tokens' và 'avg_tokens_per_case'.")
        else:
            if not _is_number(cost["total_tokens"]) or cost["total_tokens"] < 0:
                errors.append("cost.total_tokens phải là số không âm.")
            if not _is_number(cost["avg_tokens_per_case"]) or cost["avg_tokens_per_case"] < 0:
                errors.append("cost.avg_tokens_per_case phải là số không âm.")


def _validate_benchmark_results(
    results: Any,
    expected_total: int,
    summary_counts: Dict[str, Any] | None,
    errors: List[str],
    warnings: List[str],
) -> None:
    if not isinstance(results, list):
        errors.append("benchmark_results.json phải là JSON array.")
        return

    if len(results) != expected_total:
        errors.append(
            f"Số phần tử benchmark_results ({len(results)}) != metadata.total ({expected_total})."
        )

    status_counter = {"pass": 0, "fail": 0, "error": 0}
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            errors.append(f"benchmark_results[{i}] không phải object.")
            continue

        for key in ("idx", "question", "status", "latency_sec"):
            if key not in item:
                errors.append(f"benchmark_results[{i}] thiếu trường '{key}'.")

        status = item.get("status")
        if status not in status_counter:
            errors.append(f"benchmark_results[{i}].status không hợp lệ: {status}")
            continue
        status_counter[status] += 1

        if status in ("pass", "fail"):
            retrieval = item.get("retrieval")
            judge = item.get("judge")
            cost = item.get("cost")
            if not isinstance(retrieval, dict):
                errors.append(f"benchmark_results[{i}].retrieval phải là object với case pass/fail.")
            else:
                for key in ("hit_rate", "mrr", "expected_ids", "retrieved_ids", "has_ground_truth"):
                    if key not in retrieval:
                        errors.append(f"benchmark_results[{i}].retrieval thiếu '{key}'.")

            if not isinstance(judge, dict):
                errors.append(f"benchmark_results[{i}].judge phải là object với case pass/fail.")
            else:
                for key in ("final_score", "agreement_rate", "conflict_detected"):
                    if key not in judge:
                        errors.append(f"benchmark_results[{i}].judge thiếu '{key}'.")

            if not isinstance(cost, dict):
                errors.append(f"benchmark_results[{i}].cost phải là object với case pass/fail.")
            else:
                if "tokens_used" not in cost:
                    errors.append(f"benchmark_results[{i}].cost thiếu 'tokens_used'.")

    if summary_counts is not None:
        for key in ("pass", "fail", "error"):
            if summary_counts.get(key) != status_counter[key]:
                errors.append(
                    f"counts.{key} ({summary_counts.get(key)}) != số case '{key}' trong benchmark_results ({status_counter[key]})."
                )

    if status_counter["error"] > 0:
        warnings.append(f"Có {status_counter['error']} case ở trạng thái error trong benchmark_results.")


def validate_lab() -> bool:
    print("🔍 Đang kiểm tra định dạng bài nộp...")

    missing = []
    for path in REQUIRED_FILES:
        if os.path.exists(path):
            print(f"✅ Tìm thấy: {path}")
        else:
            print(f"❌ Thiếu file: {path}")
            missing.append(path)

    if missing:
        print(f"\n❌ Thiếu {len(missing)} file. Hãy bổ sung trước khi nộp bài.")
        return False

    errors: List[str] = []
    warnings: List[str] = []

    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ reports/summary.json không phải JSON hợp lệ: {e}")
        return False

    try:
        with open("reports/benchmark_results.json", "r", encoding="utf-8") as f:
            benchmark_results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ reports/benchmark_results.json không phải JSON hợp lệ: {e}")
        return False

    _validate_summary_schema(summary, errors, warnings)

    metadata = summary.get("metadata", {})
    expected_total = metadata.get("total", 0) if isinstance(metadata, dict) else 0
    summary_counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else None

    _validate_benchmark_results(benchmark_results, expected_total, summary_counts, errors, warnings)

    print("\n--- Thống kê nhanh ---")
    metrics = summary.get("metrics", {})
    print(f"Tổng số cases: {expected_total}")
    if _is_number(metrics.get("avg_score")):
        print(f"Điểm trung bình: {metrics['avg_score']:.2f}")
    if _is_number(metrics.get("hit_rate")):
        print(f"Hit Rate: {metrics['hit_rate']*100:.1f}%")
    if _is_number(metrics.get("mrr")):
        print(f"MRR: {metrics['mrr']:.4f}")
    if _is_number(metrics.get("agreement_rate")):
        print(f"Agreement Rate: {metrics['agreement_rate']*100:.1f}%")

    if warnings:
        print("\n⚠️ Cảnh báo:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\n❌ Lỗi định dạng/phù hợp:")
        for e in errors:
            print(f"- {e}")
        return False

    print("\n🚀 Bài lab đã sẵn sàng để chấm điểm!")
    return True


if __name__ == "__main__":
    ok = validate_lab()
    raise SystemExit(0 if ok else 1)
