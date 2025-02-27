from __future__ import annotations

import torch

from interpreto.attributions.perturbations.base import Perturbator


class LinearInterpolationPerturbation(Perturbator):
    """
    Perturbation using linear interpolation TODO: add docstring
    """

    def __init__(self, baseline: torch.Tensor | float | None = None, n_samples: int = 10):
        super().__init__(baseline=baseline, n_samples=n_samples)

    def perturb(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:  # TODO: test
        """
        TODO: add docstring
        """
        baseline = Perturbator.adjust_baseline(self.baseline, inputs)
        assert inputs.shape[1:] == baseline.shape
        # Shape: (1, steps, ...)
        alphas = torch.linspace(0, 1, self.n_samples, device=inputs.device).view(
            1, self.n_samples, *([1] * (inputs.dim() - 1))
        )

        # Shape: (batch_size, steps:1, *input_shape)
        inputs = inputs.unsqueeze(1)

        # Shape: (batch_size:1, steps:1, *input_shape)
        baseline = baseline.to(inputs.device).view(1, 1, *baseline.shape)

        # Perform interpolation
        interpolated = (1 - alphas) * inputs + alphas * baseline

        baseline = baseline.cpu()

        return interpolated, None
