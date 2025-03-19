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

"""
Basic standard classes for attribution methods
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch
from jaxtyping import Float

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import BasePerturbator
from interpreto.commons.generator_tools import PersistentTupleGeneratorWrapper
from interpreto.typing import ModelInput

SingleAttribution = (
    Float[torch.Tensor, "l"] | Float[torch.Tensor, "l c"] | Float[torch.Tensor, "l l_g"] | Float[torch.Tensor, "l l_t"]
)

# TODO : move this in generator tools



class AttributionOutput:
    """
    Class to store the output of an attribution method.
    """

    def __init__(
        self,
        attributions: SingleAttribution,
        elements: list[str] | torch.Tensor | None = None,
    ):
        """
        Initializes an AttributionOutput instance.

        Args:
            attributions (Iterable[SingleAttribution]): A list (n elements, with n the number of samples) of attribution score tensors:
                - `l` represents the number of elements for which attribution is computed (for NLP tasks: can be the total sequence length).
                - Shapes depend on the task:
                    - Classification (single class): `(l,)`
                    - Classification (all classes): `(l, c)`, where `c` is the number of classes.
                    - Generative models: `(l, l_g)`, where `l_g` is the length of the generated part.
                        - For non-generated elements, there are `l_g` attribution scores.
                        - For generated elements, scores are zero for previously generated tokens.
                    - Token classification: `(l, l_t)`, where `l_t` is the number of token classes. When the tokens are disturbed, l = l_t.

            elements (Iterable[list[str]] | Iterable[torch.Tensor] | None, optional): A list or tensor representing the elements for which attributions are computed.
                - These elements can be tokens, words, sentences, or tensors of size `l`.
        """
        self.attributions = attributions
        self.elements = elements


class AttributionExplainer:
    """
    Abstract class for attribution methods, gives specific types of explanations
    """

    def __init__(
        self,
        inference_wrapper: Callable,
        perturbator: BasePerturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
    ):
        self.perturbator = perturbator
        self.inference_wrapper = inference_wrapper
        self.aggregator = aggregator
        self.device = device

    @abstractmethod
    def explain(self, inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        # TODO : give more generic type for model output / target
        """
        main process of attribution method
        """
        raise NotImplementedError

    def __call__(self, inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        return self.explain(inputs, targets)


class InferenceExplainer(AttributionExplainer):
    """
    Black box model explainer
    """

    # TODO : faire ça différement
    def call_inference_wrapper(self, inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor) -> Any:
        return self.inference_wrapper.get_scores(inputs, targets)

    def explain(self, inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        """
        main process of attribution method
        """
        self.inference_wrapper.to(self.device)

        tokens = [self.perturbator.tokenizer(item, return_tensors="pt") for item in inputs]
        token_count = [len(item["input_ids"][0]) for item in tokens]
        sorted_indices = sorted(range(len(token_count)), key=lambda k: token_count[k], reverse=True)

        if targets is None:
            for t in tokens:
                t["input_ids"] = t["input_ids"].unsqueeze(1)
                t["attention_mask"] = t["attention_mask"].unsqueeze(1)
            targets = torch.stack(list(self.inference_wrapper.get_targets([tokens[i] for i in sorted_indices])))
        # Perturbation
        pert_per_input_generator = PersistentTupleGeneratorWrapper(self.perturbator.perturb(inputs[i]) for i in sorted_indices)
        scores = list(self.call_inference_wrapper(pert_per_input_generator.get_subgenerator(0), targets))
        masks = list(pert_per_input_generator.get_subgenerator(1))

        # Unsort
        unsorted_masks = [None] * len(masks)
        unsorted_scores = [None] * len(scores)
        for idx, original_idx in enumerate(sorted_indices):
            unsorted_masks[original_idx] = masks[idx]
            unsorted_scores[original_idx] = scores[idx]

        explanations = []

        for score, mask in zip(unsorted_scores, unsorted_masks, strict=True):
            explanation = self.aggregator(score, mask)
            explanations.append(explanation)

        return explanations


class GradientExplainer(InferenceExplainer):
    """
    Explainer using differentiability of model to produce explanations (integrated gradients, deeplift...)
    Can be fully constructed from a perturbation and an aggregation
    Subclasses of this explainer are mostly reductions to a specific perturbation or aggregation
    """

    def call_inference_wrapper(self, inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor) -> Any:
        return self.inference_wrapper.get_gradients(inputs, targets)