"""
Base classes for perturbations used in attribution methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable

import torch

from interpreto.typing import ModelInput


class Perturbator(ABC):
    """
    Object allowing you to perturb an input (add noise, change tokens, create progression of vectors...)
    """

    @abstractmethod
    def perturb(
        self, inputs: ModelInput | Collection[ModelInput]
    ) -> tuple[torch.Tensor] | tuple[Iterable[torch.Tensor]]:
        """
        Method to perturb an input, should return a collection of perturbed elements and their associated masks
        """


class TokenPerturbator(Perturbator):
    """
    Generic class for token modification (occlusion, words substitution...)
    """

class WordPerturbator(Perturbator):
    """
    Generic class for word-wise modification
    """

class TensorPerturbator(Perturbator):
    """
    Generic class for any tensor-wise modification
    """


class GaussianNoisePerturbator(TensorPerturbator):
    """
    Perturbator adding gaussian noise to the input tensor
    """

    def __init__(self, n_perturbations: int = 10, *, std: float = 0.1):
        self.n_perturbations = n_perturbations
        self.std = std

    def perturb(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        noise = torch.randn(self.n_perturbations, *inputs.shape, device=inputs.device) * self.std
        return inputs + noise, None
