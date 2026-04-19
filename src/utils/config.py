"""Configuration loading and merging utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def merge_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI argument overrides into the loaded config.

    Supports flat keys that map to nested config paths:
        "model" -> config["model"]["name"]
        "lr"    -> config["training"]["learning_rate"]
        "epochs" -> config["training"]["epochs"]
        "batch_size" -> config["data"]["batch_size"]
    """
    key_map = {
        "model": ("model", "name"),
        "lr": ("training", "learning_rate"),
        "epochs": ("training", "epochs"),
        "batch_size": ("data", "batch_size"),
        "dropout": ("model", "dropout"),
    }

    for cli_key, value in overrides.items():
        if value is None:
            continue
        if cli_key in key_map:
            section, param = key_map[cli_key]
            config[section][param] = value
        else:
            config[cli_key] = value

    return config


def get_device(device_config: str) -> str:
    """Resolve device string. 'auto' picks the best available."""
    import torch

    if device_config != "auto":
        return device_config

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
