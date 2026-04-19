"""Entry point for model training.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/mobilenet_v2.yaml --lr 0.0001
    python scripts/train.py --config configs/custom_cnn.yaml --epochs 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.dataset import CelebAGenderDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import get_device, load_config, merge_cli_overrides
from src.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train gender recognition model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()


def build_optimizer(model, config):
    training_cfg = config["training"]
    name = training_cfg["optimizer"]
    lr = training_cfg["learning_rate"]
    wd = training_cfg["weight_decay"]

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config):
    training_cfg = config["training"]
    name = training_cfg["scheduler"]
    epochs = training_cfg["epochs"]

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli_overrides(config, vars(args))

    set_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Using device: {device}")

    data_cfg = config["data"]
    image_size = data_cfg["image_size"]

    train_dataset = CelebAGenderDataset(
        root_dir=data_cfg["root_dir"],
        split="train",
        transform=get_train_transforms(image_size),
    )
    val_dataset = CelebAGenderDataset(
        root_dir=data_cfg["root_dir"],
        split="val",
        transform=get_eval_transforms(image_size),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    print(f"Model: {config['model']['name']}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )


if __name__ == "__main__":
    main()
