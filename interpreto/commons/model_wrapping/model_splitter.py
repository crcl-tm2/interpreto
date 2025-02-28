"""
Model splitting classes based on NNsight
"""

from __future__ import annotations

from collections import OrderedDict

from torch import nn

from interpreto.typing import LatentActivation, ModelInput


class ModelSplitterPlaceholder:
    """
    Placeholder class for model splitting
    """

    def __init__(self, model: nn.Module, split: str | list[str]):
        assert split == "input_to_latent"
        self.split = split
        assert hasattr(model, "input_to_latent") and hasattr(model, "end_model")
        self.model = model
        self.model_parts = OrderedDict(
            {
                "input_to_latent": self.model.input_to_latent,
                "end_model": self.model.end_model,
            }
        )

    def get_activations(self, inputs: ModelInput) -> dict[str, LatentActivation]:
        """
        Get activations for a given input at each layer specified by the split
        """
        return self.model_parts[self.split](inputs)
