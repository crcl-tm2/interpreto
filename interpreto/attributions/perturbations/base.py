"""
Base classes for perturbations used in attribution methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

import torch

from interpreto.typing import ModelInput, TokenEmbedding


class Perturbator(ABC):
    """
    Object allowing you to perturb an input (add noise, change tokens, create progression of vectors...)
    """
    @abstractmethod
    def perturb(self, inputs: ModelInput | Collection[ModelInput], n_samples:int=10) -> Collection[TokenEmbedding]:
        """
        Method to perturb an input, should return a collection of perturbed elements and their associated masks
        """

class TokenPerturbation(Perturbator):
    """
    Generic class for token modification (occlusion, words substitution...)
    """


class TensorPerturbation(Perturbator):
    """
    Generic class for any tensor-wise modification
    """

class GaussianNoisePerturbator(TensorPerturbation):
    """
    Perturbator adding gaussian noise to the input tensor
    """
    def __init__(self, std: float = 0.1):
        self.std = std

    def perturb(self, inputs: torch.Tensor, n_samples: int = 10) -> tuple[torch.Tensor, None]:
        noise = torch.randn(n_samples, *inputs.shape, device=inputs.device) * self.std
        return inputs + noise, None