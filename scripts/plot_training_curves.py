"""Generate paper-ready training plots from saved history JSON files.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --models custom_cnn mobilenet_v2
    python scripts/plot_training_curves.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "mobilenet_v2": "MobileNetV2",
    "xception": "Xception",
}

MODEL_COLORS = {
    "custom_cnn": "#1f77b4",
    "mobilenet_v2": "#d62728",
    "xception": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training histories")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory containing *_history.json files and where plots are saved",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models to include (e.g. custom_cnn mobilenet_v2)",
    )
    return parser.parse_args()


def get_display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def load_histories(output_dir: Path, models: list[str] | None) -> dict[str, dict[str, list[float]]]:
    histories: dict[str, dict[str, list[float]]] = {}

    for path in sorted(output_dir.glob("*_history.json")):
        model_name = path.stem.replace("_history", "")
        if models is not None and model_name not in models:
            continue
        with open(path) as f:
            histories[model_name] = json.load(f)

    if not histories:
        requested = ", ".join(models) if models else "any models"
        raise FileNotFoundError(
            f"No training history files found in {output_dir} for {requested}."
        )

    return histories


def plot_metric_comparison(
    histories: dict[str, dict[str, list[float]]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, history in histories.items():
        epochs = range(1, len(history[metric_key]) + 1)
        ax.plot(
            epochs,
            history[metric_key],
            label=get_display_name(model_name),
            marker="o",
            linewidth=2.2,
            markersize=4,
            color=MODEL_COLORS.get(model_name),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_overview(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    metrics = [
        ("train_loss", "Training Loss", "Loss"),
        ("val_loss", "Validation Loss", "Loss"),
        ("val_accuracy", "Validation Accuracy", "Score"),
        ("val_f1", "Validation F1", "Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        for model_name, history in histories.items():
            epochs = range(1, len(history[metric_key]) + 1)
            ax.plot(
                epochs,
                history[metric_key],
                label=get_display_name(model_name),
                marker="o",
                linewidth=2.0,
                markersize=3.5,
                color=MODEL_COLORS.get(model_name),
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if metric_key.startswith("val_") and metric_key != "val_loss":
            ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_train_vs_val(history: dict[str, list[float]], model_name: str, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2.0)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", linewidth=2.0)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{get_display_name(model_name)}: Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", marker="o", linewidth=2.0)
    axes[1].plot(epochs, history["val_precision"], label="Val Precision", marker="^", linewidth=2.0)
    axes[1].plot(epochs, history["val_recall"], label="Val Recall", marker="v", linewidth=2.0)
    axes[1].plot(epochs, history["val_f1"], label="Val F1", marker="s", linewidth=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title(f"{get_display_name(model_name)}: Validation Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def build_training_summary(histories: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows = []
    for model_name, history in histories.items():
        val_loss = history["val_loss"]
        val_accuracy = history["val_accuracy"]
        val_f1 = history["val_f1"]

        best_loss_epoch = min(range(len(val_loss)), key=val_loss.__getitem__) + 1
        best_acc_epoch = max(range(len(val_accuracy)), key=val_accuracy.__getitem__) + 1
        best_f1_epoch = max(range(len(val_f1)), key=val_f1.__getitem__) + 1

        rows.append(
            {
                "model": model_name,
                "epochs_ran": len(history["train_loss"]),
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": val_loss[-1],
                "final_val_accuracy": val_accuracy[-1],
                "final_val_f1": val_f1[-1],
                "best_val_loss": min(val_loss),
                "best_val_loss_epoch": best_loss_epoch,
                "best_val_accuracy": max(val_accuracy),
                "best_val_accuracy_epoch": best_acc_epoch,
                "best_val_f1": max(val_f1),
                "best_val_f1_epoch": best_f1_epoch,
            }
        )

    return pd.DataFrame(rows).set_index("model").sort_index()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    histories = load_histories(output_dir, args.models)

    plot_metric_comparison(
        histories,
        metric_key="train_loss",
        title="Training Loss by Model",
        ylabel="Loss",
        output_path=output_dir / "learning_curves_train_loss.png",
    )
    plot_metric_comparison(
        histories,
        metric_key="val_loss",
        title="Validation Loss by Model",
        ylabel="Loss",
        output_path=output_dir / "learning_curves_val_loss.png",
    )
    plot_metric_comparison(
        histories,
        metric_key="val_accuracy",
        title="Validation Accuracy by Model",
        ylabel="Accuracy",
        output_path=output_dir / "learning_curves_val_accuracy.png",
    )
    plot_metric_comparison(
        histories,
        metric_key="val_f1",
        title="Validation F1 by Model",
        ylabel="F1 Score",
        output_path=output_dir / "learning_curves_val_f1.png",
    )
    plot_overview(histories, output_dir / "learning_curves_overview.png")

    for model_name, history in histories.items():
        plot_train_vs_val(history, model_name, output_dir / f"{model_name}_train_vs_val.png")

    summary_df = build_training_summary(histories)
    summary_path = output_dir / "training_summary.csv"
    summary_df.to_csv(summary_path, float_format="%.6f")
    print(f"Saved {summary_path}")

    print("\nTraining summary:")
    print(summary_df.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
