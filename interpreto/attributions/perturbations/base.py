"""
Base classes for perturbations used in attribution methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

import torch

from interpreto.typing import ModelInput, TensorBaseline, TokenEmbedding


class Perturbator(ABC):
    """
    Object allowing you to perturb an input (add noise, change tokens, create progression of vectors...)
    """

    def __init__(
        self,
        baseline: TensorBaseline = None,
        n_samples: int = 10,
    ):
        assert isinstance(baseline, (torch.Tensor, int, float, type(None)))  # noqa: UP038
        assert isinstance(n_samples, int) and n_samples > 0
        self.baseline = baseline
        self.n_samples = n_samples

    @staticmethod
    def adjust_baseline(baseline: TensorBaseline, inputs: torch.Tensor) -> torch.Tensor:
        """
        Ensures the 'baseline' argument is correctly adjusted based on the shape of 'inputs' (PyTorch tensor).

        - If baseline is None, it is replaced with a tensor of zeros matching input.shape[1:].
        - If baseline is a float, it is broadcasted to input.shape[1:].
        - If baseline is a tensor, its shape must match input.shape[1:]; otherwise, an error is raised.

        Args:
            baseline: The baseline to adjust.
            inputs: The input to adjust the baseline for.

        Returns:
            The adjusted baseline.
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Expected 'inputs' to be a PyTorch tensor.")

        # Shape: (batch_size, *input_shape)
        input_shape = inputs.shape[1:]

        if baseline is None:
            baseline = 0

        if isinstance(baseline, (int, float)):  # noqa: UP038
            baseline = torch.full(input_shape, baseline, dtype=inputs.dtype, device=inputs.device)
        elif isinstance(baseline, torch.Tensor):
            if baseline.shape != input_shape:
                raise ValueError(f"Baseline shape {baseline.shape} does not match expected shape {input_shape}.")
            if baseline.dtype != inputs.dtype:
                raise ValueError(f"Baseline dtype {baseline.dtype} does not match expected dtype {inputs.dtype}.")
        else:
            raise TypeError("Baseline must be None, a float, or a PyTorch tensor.")

        return baseline

    @abstractmethod
    def perturb(self, item: ModelInput | Collection[ModelInput]) -> Collection[TokenEmbedding]:
        """
        Method to perturb an input, should return a collection of perturbed elements and their associated masks
        """
        perturbed_elements = ...
        masks = ...
        return perturbed_elements, masks


class TokenPerturbation(Perturbator):
    """
    Generic class for token modification (occlusion, words substitution...)
    """


class TensorPerturbation(Perturbator):
    """
    Generic class for any tensor-wise modification
    """
