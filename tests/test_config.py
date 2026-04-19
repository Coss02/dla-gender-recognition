"""Tests for config loading utilities."""

from pathlib import Path

import pytest

from src.utils.config import load_config, merge_cli_overrides


@pytest.fixture()
def sample_config(tmp_path: Path):
    """Create a minimal config YAML file."""
    config_content = """
data:
  batch_size: 64
model:
  name: "custom_cnn"
  dropout: 0.3
training:
  learning_rate: 0.001
  epochs: 30
seed: 42
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


class TestLoadConfig:
    def test_loads_valid_yaml(self, sample_config):
        config = load_config(sample_config)
        assert config["seed"] == 42
        assert config["data"]["batch_size"] == 64

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestMergeOverrides:
    def test_override_learning_rate(self, sample_config):
        config = load_config(sample_config)
        config = merge_cli_overrides(config, {"lr": 0.0001})
        assert config["training"]["learning_rate"] == 0.0001

    def test_override_model_name(self, sample_config):
        config = load_config(sample_config)
        config = merge_cli_overrides(config, {"model": "mobilenet_v2"})
        assert config["model"]["name"] == "mobilenet_v2"

    def test_none_values_are_skipped(self, sample_config):
        config = load_config(sample_config)
        config = merge_cli_overrides(config, {"lr": None, "epochs": None})
        assert config["training"]["learning_rate"] == 0.001
        assert config["training"]["epochs"] == 30
