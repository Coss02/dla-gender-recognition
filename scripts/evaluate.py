"""Entry point for model evaluation on the test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CelebAGenderDataset
from src.data.transforms import get_eval_transforms
from src.models.factory import build_model
from src.training.metrics import compute_binary_metrics, compute_confusion_matrix
from src.utils.config import get_device, load_config
from src.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate gender recognition model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Override config (default: use config saved in checkpoint)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save results")
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    """Run model on all batches, collect predictions and labels."""
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_confusion_matrix(cm, output_path):
    """Save a confusion matrix heatmap to disk."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Female", "Male"],
        yticklabels=["Female", "Male"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint["config"]

    set_seed(config["seed"])
    device = get_device(config["device"])

    data_cfg = config["data"]
    test_dataset = CelebAGenderDataset(
        root_dir=data_cfg["root_dir"],
        split="test",
        transform=get_eval_transforms(data_cfg["image_size"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Evaluating {config['model']['name']} on {len(test_dataset)} test samples")

    probs, labels = run_inference(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    metrics = compute_binary_metrics(labels, preds)
    cm = compute_confusion_matrix(labels, preds)

    print("\n--- Test Results ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    results = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "test_samples": len(test_dataset),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }
    results_path = output_dir / f"{model_name}_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_confusion_matrix(cm, output_dir / f"{model_name}_confusion_matrix.png")


if __name__ == "__main__":
    main()
