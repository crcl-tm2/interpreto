"""
Integrated Gradients method
"""

from __future__ import annotations

from typing import Any

import torch

from interpreto.attributions.aggregations import MeanAggregator
from interpreto.attributions.base import GradientExplainer
from interpreto.attributions.perturbations import LinearInterpolationPerturbation
from interpreto.commons.model_wrapping.inference_wrapper import ClassificationInferenceWrapper


class IntegratedGradients(GradientExplainer):
    """
    Integrated Gradients method
    """

    def __init__(
        self,
        model: Any,
        batch_size: int,
        device: torch.device | None = None,
        n_interpolations: int = 10,
        baseline: torch.Tensor | float | None = None,
    ):
        super().__init__(
            perturbation=LinearInterpolationPerturbation(baseline=baseline, n_perturbations=n_interpolations),
            inference_wrapper=ClassificationInferenceWrapper(model, batch_size=batch_size, device=device),
            aggregation=MeanAggregator(),  # TODO: check if we need a trapezoidal mean
            batch_size=batch_size,
            device=device,
        )
