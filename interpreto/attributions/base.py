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

import itertools
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from jaxtyping import Float

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import BasePerturbator
from interpreto.commons.generator_tools import PersistentTupleGeneratorWrapper
from interpreto.commons.granularity import GranularityLevel
from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper
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
        inference_wrapper: InferenceWrapper,
        perturbator: BasePerturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity_level: GranularityLevel = GranularityLevel.DEFAULT,
    ):
        self.perturbator = perturbator
        self.inference_wrapper = inference_wrapper
        self.aggregator = aggregator
        self.device = device
        self.granularity_level = granularity_level

    def process_model_inputs(self, model_inputs):
        if isinstance(model_inputs, str):
            return [self.perturbator.tokenizer(model_inputs, return_tensors="pt")]
        if isinstance(model_inputs, Mapping):
            if model_inputs["attention_mask"].shape[0] > 1:
                n = next(iter(model_inputs.values())).shape[0]
                return [{key: value[i] for key, value in model_inputs.items()} for i in range(n)]
            return [model_inputs]
        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))
        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    def process_targets_generation(self, targets):
        if isinstance(targets, str):
            return [self.perturbator.tokenizer(targets, return_tensors="pt")["input_ids"]]
        if isinstance(targets, Mapping):
            targets = targets["input_ids"]
            if targets.shape[0] > 1:
                return list(targets.split(1, dim=0))
            return [targets]
        if isinstance(targets, Iterable):
            return list(itertools.chain(*[self.process_targets(item) for item in targets]))
        raise ValueError(
            f"type {type(targets)} not supported for method process_targets in class {self.__class__.__name__}"
        )

    @abstractmethod
    def explain(self, model_inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        """
        main process of attribution method
        """
        raise NotImplementedError

    def __call__(self, model_inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        return self.explain(model_inputs, targets)


class ClassificationInferenceExplainer(AttributionExplainer):
    """
    Black box model explainer
    """

    def explain(self, model_inputs: ModelInput, targets: torch.Tensor | None = None) -> Any:
        """
        main process of attribution method
        """

        self.inference_wrapper.to(self.device)
        model_inputs = self.process_model_inputs(model_inputs)

        # token_count = [len(item["input_ids"][0]) for item in model_inputs]
        # sorted_indices = sorted(range(len(token_count)), key=lambda k: token_count[k], reverse=True)

        # Reference inference
        # logits = torch.stack(list(self.inference_wrapper.get_logits(model_inputs[i] for i in sorted_indices)))
        logits = torch.stack(self.inference_wrapper.get_logits(model_inputs))

        if targets is None:
            targets = logits.argmax(dim=-1)

        logits = logits.gather(-1, targets.unsqueeze(1))

        # Perturbation
        # pert_per_input_generator = PersistentTupleGeneratorWrapper(
        #     self.perturbator.perturb(model_inputs[i]) for i in sorted_indices
        # )
        pert_per_input_generator = PersistentTupleGeneratorWrapper(self.perturbator.perturb(model_inputs))

        target_logits = list(
            self.inference_wrapper.get_target_logits(pert_per_input_generator.get_subgenerator(0), targets)
        )

        scores = [logits[index] - target_logit for index, target_logit in enumerate(target_logits)]

        masks = list(pert_per_input_generator.get_subgenerator(1))

        # Unsort
        # unsorted_masks = [None] * len(masks)
        # unsorted_scores = [None] * len(scores)
        # for idx, original_idx in enumerate(sorted_indices):
        #     unsorted_masks[original_idx] = masks[idx]
        #     unsorted_scores[original_idx] = scores[idx]

        # contributions = [
        #     self.aggregator(score.unsqueeze(0), mask).squeeze(0)
        #     for score, mask in zip(unsorted_scores, unsorted_masks, strict=True)
        # ]
        contributions = [
            self.aggregator(score.unsqueeze(0), mask).squeeze(0) for score, mask in zip(scores, masks, strict=True)
        ]

        model_inputs = [self.perturbator.tokenizer(item, return_tensors="pt") for item in model_inputs]

        decompositions = [GranularityLevel.get_decomposition(t, self.granularity_level) for t in model_inputs]
        return [
            AttributionOutput(
                c, [self.perturbator.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in d[0]]
            )
            for c, d in zip(contributions, decompositions, strict=True)
        ]


class GenerationInferenceExplainer(AttributionExplainer):
    """
    Inference based explainer
    """

    def explain(self, model_inputs: ModelInput, targets: ModelInput | None = None) -> Any:
        """
        main process of attribution method
        """
        self.inference_wrapper.to(self.device)
        model_inputs = self.process_model_inputs(model_inputs)

        if targets is None:
            model_inputs_to_explain, targets = self.inference_wrapper.get_inputs_to_explain_and_targets(model_inputs)
        else:
            targets = self.process_targets_generation(targets)
            model_inputs_to_explain = []
            for model_input, target in zip(model_inputs, targets, strict=True):
                model_input = self.inference_wrapper.embed(model_input)
                with torch.no_grad():
                    target_embed = self.inference_wrapper.model.get_input_embeddings()(target)
                model_inputs_to_explain.append(
                    {
                        "inputs_embeds": torch.cat([model_input["inputs_embeds"], target_embed], dim=1),
                        "attention_mask": torch.cat([model_input["attention_mask"], torch.ones_like(target)], dim=1),
                    }
                )

        # Perturbation
        pert_per_input_generator = PersistentTupleGeneratorWrapper(
            self.perturbator.perturb(model_input) for model_input in model_inputs_to_explain
        )

        targeted_logits = list(
            self.inference_wrapper.get_targeted_logits(pert_per_input_generator.get_subgenerator(0), targets)
        )

        masks = list(pert_per_input_generator.get_subgenerator(1))

        contributions = [
            self.aggregator(targeted_logit.unsqueeze(0), mask).squeeze(0)
            for targeted_logit, mask in zip(targeted_logits, masks, strict=True)
        ]

        decompositions = [
            GranularityLevel.get_decomposition(t, self.granularity_level) for t in model_inputs_to_explain
        ]
        return [
            AttributionOutput(
                c, [self.perturbator.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in d[0]]
            )
            for c, d in zip(contributions, decompositions, strict=True)
        ]


class GenerationGradientExplainer(AttributionExplainer):
    # TODO: a faire
    """
    Explainer using differentiability of model to produce explanations (integrated gradients, deeplift...)
    Can be fully constructed from a perturbation and an aggregation
    Subclasses of this explainer are mostly reductions to a specific perturbation or aggregation
    """

    def explain(self, model_inputs: ModelInput, targets: ModelInput | None = None) -> Any:
        """
        main process of attribution method
        """
        self.inference_wrapper.to(self.device)
        model_inputs = self.process_model_inputs(model_inputs)

        if targets is None:
            model_inputs_to_explain, targets = self.inference_wrapper.get_inputs_to_explain_and_targets(model_inputs)
        else:
            targets = self.process_targets_generation(targets)
            model_inputs_to_explain = []
            for model_input, target in zip(model_inputs, targets, strict=True):
                model_input = self.inference_wrapper.embed(model_input)
                with torch.no_grad():
                    target_embed = self.inference_wrapper.model.get_input_embeddings()(target)
                model_inputs_to_explain.append(
                    {
                        "inputs_embeds": torch.cat([model_input["inputs_embeds"], target_embed], dim=1),
                        "attention_mask": torch.cat([model_input["attention_mask"], torch.ones_like(target)], dim=1),
                    }
                )

        # Perturbation
        pert_per_input_generator = PersistentTupleGeneratorWrapper(
            self.perturbator.perturb(model_input) for model_input in model_inputs_to_explain
        )

        targeted_logits = list(
            self.inference_wrapper.get_targeted_logits(pert_per_input_generator.get_subgenerator(0), targets)
        )

        masks = list(pert_per_input_generator.get_subgenerator(1))

        contributions = [
            self.aggregator(targeted_logit.unsqueeze(0), mask).squeeze(0)
            for targeted_logit, mask in zip(targeted_logits, masks, strict=True)
        ]

        decompositions = [
            GranularityLevel.get_decomposition(t, self.granularity_level) for t in model_inputs_to_explain
        ]
        return [
            AttributionOutput(
                c, [self.perturbator.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in d[0]]
            )
            for c, d in zip(contributions, decompositions, strict=True)
        ]
