"""
Basic standard classes for attribution methods
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import Perturbator
from interpreto.typing import ModelInput


class AttributionExplainer:
    """
    Abstract class for attribution methods, gives specific types of explanations
    """

    def __init__(
        self,
        inference_wrapper: Callable,
        batch_size: int,
        perturbation: Perturbator | None = None,
        aggregation: Aggregator | None = None,
        device: torch.device | None = None,
    ):
        self.perturbation = perturbation
        self.inference_wrapper = inference_wrapper
        self.aggregation = aggregation
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def explain(self, item: ModelInput) -> Any:
        """
        main process of attribution method
        """
        raise NotImplementedError

    def __call__(self, item: ModelInput) -> Any:
        return self.explain(item)


class GradientExplainer(AttributionExplainer):
    """
    Explainer using differentiability of model to produce explanations (integrated gradients, deeplift...)
    Can be fully constructed from a perturbation and an aggregation
    Subclasses of this explainer are mostly reductions to a specific perturbation or aggregation
    """

    def explain(self, item: ModelInput) -> Any:
        """
        main process of attribution method
        """
        embeddings, _ = self.perturbation.perturb(item)

        self.inference_wrapper.to(self.device)
        results = self.inference_wrapper.batch_gradients(embeddings, flatten=True)
        self.inference_wrapper.cpu()  # TODO: check if we need to do this

        explanation = self.aggregation(results, _)

        return explanation


class InferenceExplainer(AttributionExplainer):
    """
    Black box model explainer
    """
