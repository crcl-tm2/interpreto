"""
Basic standard classes for attribution methods
"""
from __future__ import annotations
from abc import ABC
from typing import Optional, Callable, Any
from interpreto.typing import ModelInput
from interpreto.attributions.perturbations.base import Perturbator
from interpreto.attributions.aggregations.base import Aggregator

class AttributionExplainer(ABC):
    """
    Abstract class for attribution methods, gives specific types of explainations
    """
    def __init__(self, perturbation:Optional[Perturbator]=None,
                                inference_wrapper:Optional[Callable]=None,
                                aggregation:Optional[Aggregator]=None):
        self.perturbation = perturbation
        self.inference_wrapper = inference_wrapper
        self.aggregation = aggregation

    def explain(self, item:ModelInput)->Any:
        """
        main process of attribution method
        """
        embeddings, mask = self.perturbation.perturbate(item)
        results = self.inference_wrapper(embeddings)
        explaination = self.aggregation(results, mask)
        return explaination

    def __call__(self, item:ModelInput)->Any:
        return self.explain(item)

class GradientExplainer(AttributionExplainer):
    """
        Explainer using differentiability of model to produce explainations (integrated gradients, deeplift...)
        Can be fully constructed from a perturbation and an aggregation
        Subclasses of this explainer are mostly reductions to a specific perturbation or aggregation
    """

class InferenceExplainer(AttributionExplainer):
    """
    Black box model explainer
    """