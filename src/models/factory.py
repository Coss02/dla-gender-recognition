"""Model factory — builds any supported model by name.

Adding a new model only requires registering it in MODEL_REGISTRY
and implementing a builder function that returns an nn.Module.
"""

import torch.nn as nn


MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class in the factory."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(config: dict) -> nn.Module:
    """Instantiate a model from the config dict.

    Looks up config["model"]["name"] in the registry.
    Importing model modules triggers their @register_model decorators.
    """
    # Trigger registration by importing model modules
    import src.models.custom_cnn  # noqa: F401
    import src.models.finetuned   # noqa: F401

    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(config["model"])
