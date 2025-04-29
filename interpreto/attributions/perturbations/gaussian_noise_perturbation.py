from __future__ import annotations

import torch

from interpreto.attributions.perturbations.base import Perturbator
from interpreto.typing import TensorMapping


class GaussianNoisePerturbator(Perturbator):
    """
    Perturbator adding gaussian noise to the input tensor
    """

    __slots__ = ("n_perturbations", "std")

    def __init__(self, inputs_embedder: torch.nn.Module | None = None, n_perturbations: int = 10, *, std: float = 0.1):
        super().__init__(inputs_embedder)
        self.n_perturbations = n_perturbations
        self.std = std

    def perturb_embeds(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        model_inputs["inputs_embeds"] = model_inputs["inputs_embeds"].repeat(self.n_perturbations, 1, 1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(self.n_perturbations, 1, 1)

        # add noise
        model_inputs["inputs_embeds"] += torch.randn_like(model_inputs["inputs_embeds"]) * self.std

        return model_inputs, None  # return noise ? noise.bool().long() ?
