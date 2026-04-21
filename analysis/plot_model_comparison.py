import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "avg_score",
    "hit_rate",
    "mrr",
    "agreement_rate",
    "conflict_rate",
    "error_rate",
]

METRIC_LABELS = {
    "avg_score": "Avg Score",
    "hit_rate": "Hit Rate",
    "mrr": "MRR",
    "agreement_rate": "Agreement",
    "conflict_rate": "Conflict",
    "error_rate": "Error",
}


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_compare_data(summary: dict) -> tuple[str, str, list[float], list[float]]:
    regression = summary.get("regression", {})
    if not regression:
        raise ValueError("summary.json không có field 'regression'. Hãy chạy lại main.py trước.")

    v1 = regression.get("v1", {})
    v2 = regression.get("v2", {})
    if not v1 or not v2:
        raise ValueError("summary.json thiếu 'regression.v1' hoặc 'regression.v2'.")

    v1_values = [float(v1.get(m, 0.0)) for m in METRICS]
    v2_values = [float(v2.get(m, 0.0)) for m in METRICS]

    v1_name = summary.get("regression", {}).get("v1_name", "V1")
    v2_name = summary.get("metadata", {}).get("version", "V2")
    return v1_name, v2_name, v1_values, v2_values


def plot(summary_path: Path, output_path: Path) -> None:
    summary = _load_summary(summary_path)
    v1_name, v2_name, v1_values, v2_values = _extract_compare_data(summary)

    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, v1_values, width, label=v1_name, color="#1f77b4")
    bars2 = ax.bar(x + width / 2, v2_values, width, label=v2_name, color="#ff7f0e")

    ax.set_title("Model Comparison (V1 vs V2)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=20, ha="right")
    ax.set_ylim(0, max(max(v1_values), max(v2_values)) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    deltas = [v2 - v1 for v1, v2 in zip(v1_values, v2_values)]
    for i, delta in enumerate(deltas):
        y = max(v1_values[i], v2_values[i]) + 0.02
        ax.text(
            x[i],
            y,
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2d2d2d",
        )

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model comparison chart from reports/summary.json")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/summary.json"),
        help="Path to summary.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/model_comparison.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    plot(args.summary, args.out)
    print(f"Saved chart to: {args.out}")


if __name__ == "__main__":
    main()
