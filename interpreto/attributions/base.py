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
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
    ):
        self.perturbator = perturbator
        self.inference_wrapper = inference_wrapper
        self.aggregator = aggregator
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def explain(self, inputs: ModelInput, target: torch.Tensor | None = None) -> Any:
        # TODO : give more generic type for model output / target
        """
        main process of attribution method
        """
        raise NotImplementedError

    def __call__(self, inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        return self.explain(inputs, targets)


class GradientExplainer(AttributionExplainer):
    """
    Explainer using differentiability of model to produce explanations (integrated gradients, deeplift...)
    Can be fully constructed from a perturbation and an aggregation
    Subclasses of this explainer are mostly reductions to a specific perturbation or aggregation
    """

    def explain(self, inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        """
        main process of attribution method
        """
        embeddings, _ = self.perturbator.perturb(inputs)

        self.inference_wrapper.to(self.device)

        if targets is None:
            with torch.no_grad():
                targets = self.inference_wrapper.call_model(inputs)
        # repeat target along the p dimension
        targets = targets.unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        results = self.inference_wrapper.batch_gradients(embeddings, targets, flatten=True)
        self.inference_wrapper.cpu()  # TODO: check if we need to do this

        explanation = self.aggregator(results, _)

        return explanation


class InferenceExplainer(AttributionExplainer):
    """
    Black box model explainer
    """

    def explain(self, inputs: ModelInput, targets: torch.Tensor | None=None) -> Any:
        """
        main process of attribution method
        """
        embeddings, mask = self.perturbator.perturb(inputs)

        print(embeddings.shape, mask.shape)

        # embeddings.shape : (n, p, l, d)
        # target.shape : (n, o)

        self.inference_wrapper.to(self.device)

        if targets is None:
            with torch.no_grad():
                # Put this in NLP_explainer_mixin

                if isinstance(inputs, torch.Tensor):
                    targets = self.inference_wrapper.call_model(inputs)
                else:
                    tokens = self.perturbator.tokenizer(inputs,
                            truncation=True,
                            return_tensors='pt',
                            padding="max_length",
                            max_length=512,
                            return_offsets_mapping=True)
                    target_embeddings = self.inference_wrapper.model.get_input_embeddings()(tokens["input_ids"])
                    targets = self.inference_wrapper.call_model(target_embeddings)
        # repeat target along the p dimension
        targets = targets.unsqueeze(1).repeat(1, embeddings.shape[1], 1)

        # TODO : remake inference with adaptation to mask
        results = self.inference_wrapper.batch_inference(embeddings, targets, flatten=True)
        self.inference_wrapper.cpu()  # TODO: check if we need to do this

        explanation = self.aggregator(results, mask)

        return explanation
