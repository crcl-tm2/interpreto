# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch

from interpreto.attributions.perturbations.base import Perturbator
from interpreto.typing import TensorBaseline


class LinearInterpolationPerturbator(Perturbator):
    """
    Perturbation using linear interpolation between a reference point (baseline) and the input.
    """

    def __init__(self, baseline: TensorBaseline = None, n_perturbations: int = 10):
        """
        Initializes the LinearInterpolationPerturbation instance.

        Args:
            baseline (TensorBaseline, optional): The baseline value for the perturbation.
                It can be a torch.Tensor, int, float, or None. Defaults to None.

        Raises:
            AssertionError: If the baseline is not a torch.Tensor, int, float, or None.
        """
        assert isinstance(baseline, (torch.Tensor, int, float, type(None)))  # noqa: UP038
        self.n_perturbations = n_perturbations
        self.baseline = baseline

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

    def perturb(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Generates perturbed samples by performing linear interpolation between the input tensor and the baseline tensor.

        Args:
            inputs (torch.Tensor): The input tensor to be perturbed.
            n_samples (int, optional): The number of interpolation samples to generate. Defaults to 10.

        Returns:
            As no mask is required to understand perturbation, the second returned element is None.
            tuple[torch.Tensor, None]: A tuple containing the interpolated tensor and None.
        """
        baseline = self.adjust_baseline(self.baseline, inputs)
        assert inputs.shape[1:] == baseline.shape
        # Shape: (1, steps, ...)
        alphas = torch.linspace(0, 1, self.n_perturbations, device=inputs.device).view(
            1, self.n_perturbations, *([1] * (inputs.dim() - 1))
        )

        # Shape: (batch_size, steps:1, *input_shape)
        inputs = inputs.unsqueeze(1)

        # Shape: (batch_size:1, steps:1, *input_shape)
        baseline = baseline.to(inputs.device).view(1, 1, *baseline.shape)

        # Perform interpolation
        interpolated = (1 - alphas) * inputs + alphas * baseline

        baseline = baseline.cpu()

        return interpolated, None
