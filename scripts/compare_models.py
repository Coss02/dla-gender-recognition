"""Compare evaluation results across all trained models.

Reads JSON result files produced by evaluate.py and generates
a comparative summary table and a bar chart of key metrics.

Usage:
    python scripts/compare_models.py --results-dir outputs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_dir: Path) -> list[dict]:
    """Load all *_test_results.json files from the given directory."""
    results = []
    for path in sorted(results_dir.glob("*_test_results.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build a DataFrame comparing metrics across models."""
    rows = []
    for r in results:
        row = {"model": r["model"]}
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart comparing models on key metrics."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    plot_df = df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot.bar(ax=ax, rot=0, width=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument("--results-dir", type=str, default="outputs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No result files found in {results_dir}/")
        print("Run evaluate.py for each model first.")
        return

    df = build_comparison_table(results)

    print("\n=== Model Comparison (Test Set) ===\n")
    print(df.to_string(float_format="%.4f"))
    print()

    csv_path = results_dir / "model_comparison.csv"
    df.to_csv(csv_path, float_format="%.4f")
    print(f"Table saved to {csv_path}")

    plot_comparison(df, results_dir / "model_comparison.png")


if __name__ == "__main__":
    main()
