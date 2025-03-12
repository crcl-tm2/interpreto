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
from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Float

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import BasePerturbator
from interpreto.typing import ModelInput

SingleAttribution = (
    Float[torch.Tensor, "l"]
    | Float[torch.Tensor, "l c"]
    | Float[torch.Tensor, "l l_g"]
    | Float[torch.Tensor, "l l_t"]
)


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

    def explain(self, inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        """
        main process of attribution method
        """
        self.inference_wrapper.to(self.device)

        tokens = [self.perturbator.tokenizer.encode(item) for item in inputs]
        token_count = [len(item) for item in tokens]

        if targets is None:
            # split in batchs
            chunks = [tokens[i : i + self.inference_wrapper.batch_size] for i in range(0, len(tokens), self.inference_wrapper.batch_size)]
            predictions = []
            with torch.no_grad():
                for chunk in chunks:
                    padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(c) for c in chunk], batch_first=True, padding_value=self.perturbator.tokenizer.pad_token_id)
                    attention_mask = (padded_sequences != self.perturbator.tokenizer.pad_token_id).int()
                    predictions += [self.inference_wrapper.call_model({"input_ids": padded_sequences, "attention_mask": attention_mask})]
                targets = torch.cat(predictions, dim=0)

        sorted_indices = sorted(range(len(token_count)), key=lambda k: token_count[k], reverse=True)
        sorted_inputs = [inputs[i] for i in sorted_indices]

        pert_per_input_generator = (self.perturbator.perturb(item) for item in sorted_inputs)

        scores, masks = zip(*self.inference_wrapper.inference(pert_per_input_generator, targets), strict=True)

        # Unsort
        unsorted_masks = [None] * len(masks)
        unsorted_scores = [None] * len(scores)
        for idx, original_idx in enumerate(sorted_indices):
            unsorted_masks[original_idx] = masks[idx]
            unsorted_scores[original_idx] = scores[idx]

        explanations = []

        for score, mask in zip(unsorted_scores, unsorted_masks, strict=True):
            # TODO : this line must be done in inference wrapper
            #results = torch.sum(score * target.unsqueeze(0).repeat(score.shape[0], 1), dim=-1)
            explanation = self.aggregator(score.unsqueeze(0), mask)
            explanations.append(explanation)

        return explanations
