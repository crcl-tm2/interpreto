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
from collections.abc import Iterable, MutableMapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from jaxtyping import Float
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import Perturbator
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import GranularityLevel
from interpreto.commons.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper
from interpreto.commons.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper
from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper
from interpreto.typing import Generated_Target, ModelInputs, TensorMapping

SingleAttribution = (
    Float[torch.Tensor, "l"] | Float[torch.Tensor, "l c"] | Float[torch.Tensor, "l l_g"] | Float[torch.Tensor, "l l_t"]
)


class AttributionOutput:
    """
    Class to store the output of an attribution method.
    """

    __slots__ = ("attributions", "elements")

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
                    - Generative models: `(l_g, l)`, where `l_g` is the length of the generated part.
                        - For non-generated elements, there are `l_g` attribution scores.
                        - For generated elements, scores are zero for previously generated tokens.
                    - Token classification: `(l, l_t)`, where `l_t` is the number of token classes. When the tokens are disturbed, l = l_t.
            elements (Iterable[list[str]] | Iterable[torch.Tensor] | None, optional): A list or tensor representing the elements for which attributions are computed.
                - These elements can be tokens, words, sentences, or tensors of size `l`.
        """
        self.attributions = attributions
        self.elements = elements

    def __repr__(self):
        return f"AttributionOutput(attributions={repr(self.attributions)}, elements={repr(self.elements)})"

    def __str__(self):
        return f"AttributionOutput(attributions={self.attributions}, elements={self.elements})"


class AttributionExplainer:
    """
    Abstract base class for attribution explainers.

    This class defines a common interface and helper methods used by various attribution explainers.
    Subclasses must implement the abstract method 'explain'.
    """

    _associated_inference_wrapper = InferenceWrapper
    use_gradient: bool

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        # inference_wrapper: InferenceWrapper,
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity_level: GranularityLevel = GranularityLevel.DEFAULT,
    ) -> None:
        """
        Initializes the AttributionExplainer.

        Args:
            TODO : update docstring
            perturbator (Perturbator, optional): An instance for generating input perturbations.
                Defaults to a Perturbator if not provided.
            aggregator (Aggregator, optional): An instance used to aggregate computed attribution scores.
            device (torch.device, optional): The device on which computations will be performed.
            granularity_level (GranularityLevel): The level of granularity for the explanation (e.g., token, word, sentence).
        """
        self.tokenizer = tokenizer
        self.inference_wrapper = self._associated_inference_wrapper(model, batch_size=batch_size, device=device)
        self.perturbator = perturbator or Perturbator()
        self.aggregator = aggregator or Aggregator()
        self.granularity_level = granularity_level

        # TODO : check this line, eventually move it
        self.inference_wrapper.pad_token_id = self.tokenizer.pad_token_id

    def get_scores(self, model_inputs: Iterable[TensorMapping], targets: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        Computes scores for the given perturbations and targets.

        Args:
            pert_generator (Iterable[TensorMapping]): An iterable of perturbed model inputs.
            targets (torch.Tensor): The target classes or tokens.

        Returns:
            Iterable[torch.Tensor]: The computed scores.
        """
        if self.use_gradient:
            return self.inference_wrapper.get_gradients(model_inputs, targets)
        with torch.no_grad():
            return self.inference_wrapper.get_targeted_logits(model_inputs, targets)
    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is located.
        """
        return self.inference_wrapper.device

    @device.setter
    def device(self, device: torch.device) -> None:
        """
        Sets the device on which the model is located.
        """
        self.inference_wrapper.device = device

    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The device to which the model should be moved.
        """
        self.inference_wrapper.to(device)

    def process_model_inputs(self, model_inputs: ModelInputs) -> list[TensorMapping]:
        """
        Processes and standardizes model inputs into a list of dictionaries compatible with the model.

        This method handles various input types:
            - If a string is provided, it tokenizes the string and returns a list containing one mapping.
            - If a mapping is provided with a batch (multiple samples), it splits the batch into individual mappings.
            - If an iterable is provided, it processes each item recursively.

        Args:
            model_inputs (str, MutableMapping, or Iterable): The raw model inputs.

        Returns:
            List[MutableMapping]: A list of processed model input mappings.

        Raises:
            ValueError: If the type of model_inputs is not supported.
        """
        if isinstance(model_inputs, str):
            return [
                self.tokenizer(
                    model_inputs, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True
                )
            ]
        if isinstance(model_inputs, MutableMapping):
            return [
                {key: value[i].unsqueeze(0) for key, value in model_inputs.items()}
                for i in range(model_inputs["attention_mask"].shape[0])
            ]
        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))
        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    def process_inputs_to_explain_and_targets(
        self, model_inputs: Iterable[TensorMapping], targets: Sequence[torch.Tensor] | None, **model_kwargs: Any
    ) -> tuple[list[TensorMapping], Sequence[torch.Tensor]]:
        # TODO : update docstring and add error message
        raise NotImplementedError()

    def explain(
        self, model_inputs: ModelInputs, targets: Generated_Target = None, **model_kwargs: Any
    ) -> Iterable[AttributionOutput]:
        """
        Computes attributions for generative models.

        Process:
            1. Move the model to the designated device.
            2. Process and standardize the model inputs.
            3. Create the tokenizer's pad token if not already set and add it to the inference wrapper.
            4. If targets are not provided, create them. Otherwise, for each input-target pair, process them.
            5. Generate perturbations for the constructed inputs.
            6. Compute scores using either gradients (if use_gradient is True) or targeted logits.
            7. Aggregate the scores to obtain contribution values.
            8. Decompose the inputs based on the desired granularity and decode tokens.

        Args:
            model_inputs (ModelInputs): Raw inputs for the generative model.
            targets (ModelInputs, optional): Target texts or tokens for which explanations are desired.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        model_inputs = self.process_model_inputs(model_inputs)

        # give pad token id to the inference wrapper
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_wrapper.pad_token_id = self.tokenizer.pad_token_id

        model_inputs_to_explain, targets = self.process_inputs_to_explain_and_targets(
            model_inputs, targets, **model_kwargs
        )

        # TODO : change this line to avoid copying the model inputs
        pert_generator, mask_generator = split_iterator(
            self.perturbator.perturb(m) for m in deepcopy(model_inputs_to_explain)
        )

        scores = self.get_scores(pert_generator, targets.to(self.device))

        # Aggregate the scores using the aggregator to obtain contribution values.

        # TODO : check if we need to add a squeeze(0) here (in generation version we have but not in classification)
        # contributions = [
        #     self.aggregator(score.unsqueeze(0), mask).squeeze(0)
        #     for score, mask in zip(scores, masks, strict=True)  # generation version
        # ]
        contributions = [
            self.aggregator(score, mask.to(self.device)).squeeze(0)
            for score, mask in zip(scores, mask_generator, strict=True)
        ]  # classification version

        # Decompose each input for the desired granularity level
        decompositions = [
            GranularityLevel.get_decomposition(t, self.granularity_level) for t in model_inputs_to_explain
        ]

        # Create and return AttributionOutput objects with the contributions and decoded token sequences:
        return [
            AttributionOutput(
                c,
                [
                    self.tokenizer.decode(
                        token_ids, skip_special_tokens=self.granularity_level is not GranularityLevel.ALL_TOKENS
                    )
                    for token_ids in d[0]
                ],
            )
            for c, d in zip(contributions, decompositions, strict=True)
        ]

    def __call__(
        self, model_inputs: ModelInputs, targets: ModelInputs | torch.Tensor | None = None
    ) -> Iterable[AttributionOutput]:
        """
        Enables the explainer instance to be called as a function.

        Args:
            model_inputs (ModelInputs): The inputs to the model.
            targets (torch.Tensor, optional): The target classes or tokens.

        Returns:
            Any: The computed attributions.
        """
        return self.explain(model_inputs, targets)


class ClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for classification models
    """

    _associated_inference_wrapper = ClassificationInferenceWrapper

    def process_inputs_to_explain_and_targets(
        self, model_inputs: Iterable[TensorMapping], targets: torch.Tensor | None
    ) -> tuple[Iterable[TensorMapping], torch.Tensor]:
        logits = torch.stack(list(self.inference_wrapper.get_logits(deepcopy(model_inputs))))
        if targets is None:
            targets = logits.argmax(dim=-1)

        # TODO : change call to process_target
        targets = ClassificationInferenceWrapper.process_target(targets, logits.shape[:-1])
        return model_inputs, targets


class GenerationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for generation models
    """

    _associated_inference_wrapper = GenerationInferenceWrapper

    def process_targets(self, targets: Generated_Target) -> list[torch.Tensor]:
        """
        Processes the target inputs for generative models into a standardized format.

        This function handles various input types for targets (string, MutableMapping, or Iterable)
        and converts them into a list of tensors containing token IDs.

        Args:
            targets (str, MutableMapping, torch.Tensor, or Iterable): The target texts or tokens.

        Returns:
            List[torch.Tensor]: A list of tensors representing the target token IDs.

        Raises:
            ValueError: If the target type is not supported.
        """
        if isinstance(targets, str):
            return [self.tokenizer(targets, return_tensors="pt")["input_ids"]]
        if isinstance(targets, MutableMapping):
            targets = targets["input_ids"]
            if targets.shape[0] > 1:
                return list(targets.split(1, dim=0))
            return [targets]
        if isinstance(targets, torch.Tensor):
            return [targets]
        if isinstance(targets, Iterable):
            return list(itertools.chain(*[self.process_targets(item) for item in targets]))
        raise ValueError(
            f"type {type(targets)} not supported for method process_targets in class {self.__class__.__name__}"
        )

    def process_inputs_to_explain_and_targets(
        self, model_inputs: ModelInputs, targets: Generated_Target, **model_kwargs: dict[str, Any]
    ) -> tuple[list[TensorMapping], Sequence[torch.Tensor]]:
        """
        Processes the inputs and targets for the generative model.
        If targets are not provided, create them with model_inputs_to_explain. Otherwise, for each input-target pair:
            a. Embed the input.
            b. Embed the target and concatenate with the input embeddings.
            c. Construct a new input mapping that includes both embeddings.

        Args:
            model_inputs (ModelInputs): The raw inputs for the generative model.
            targets (Generated_Target): The target texts or tokens for which explanations are desired.
            model_kwargs (dict): Additional arguments for the generation process.

        Returns:
            tuple: A tuple containing a list of processed model inputs and a list of processed targets.
        """
        if targets is None:
            model_inputs_to_explain_basic, targets = self.inference_wrapper.get_inputs_to_explain_and_targets(
                model_inputs, **model_kwargs
            )
            print(model_inputs_to_explain_basic)
        else:
            targets = self.process_targets(targets)
            model_inputs_to_explain_basic = []
            for model_input, target in zip(model_inputs, targets, strict=True):
                # embed_model_input = self.inference_wrapper.embed(model_input)
                # with torch.no_grad():
                #    target_embed = self.inference_wrapper.model.get_input_embeddings()(target)
                model_inputs_to_explain_basic.append(
                    {
                        # "inputs_embeds": torch.cat([embed_model_input["inputs_embeds"], target_embed], dim=1),
                        "input_ids": torch.cat([model_input["input_ids"], target], dim=1),
                        "attention_mask": torch.cat([model_input["attention_mask"], torch.ones_like(target)], dim=1),
                    }
                )
        # Add offsets mapping and special tokens mask:
        model_inputs_to_explain_text = [
            self.tokenizer.decode(elem["input_ids"][0]) for elem in model_inputs_to_explain_basic
        ]
        model_inputs_to_explain_without_embed = [
            self.tokenizer(
                [model_inputs_to_explain_text],
                return_tensors="pt",
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            for model_inputs_to_explain_text in model_inputs_to_explain_text
        ]
        # Add inputs_embeds:
        model_inputs_to_explain = [
            self.inference_wrapper.embed(elem) for elem in model_inputs_to_explain_without_embed
        ]

        # Decompose each input for the desired granularity level.
        # TODO: move this in a better place
        if self.granularity_level == GranularityLevel.TOKEN:
            self.granularity_level = GranularityLevel.ALL_TOKENS  # equal for generative models

        return model_inputs_to_explain, targets


class FactoryGeneratedMeta(type):
    """
    Metaclass to distinguish classes generated by the MultitaskExplainerMixin.
    """


class MultitaskExplainerMixin(AttributionExplainer):
    """
    Mixin class to generate the appropriate Explainer based on the model type.
    """

    def __new__(cls, model: PreTrainedModel, *args: Any, **kwargs: Any) -> AttributionExplainer:
        if isinstance(cls, FactoryGeneratedMeta):
            return super().__new__(cls)  # type: ignore
        if model.__class__.__name__.endswith("ForSequenceClassification"):
            t = FactoryGeneratedMeta("Classification" + cls.__name__, (cls, ClassificationAttributionExplainer), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
            t = FactoryGeneratedMeta("Generation" + cls.__name__, (cls, GenerationAttributionExplainer), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        raise NotImplementedError(
            "Model type not supported for Explainer. Use a ModelForSequenceClassification, a ModelForCausalLM model or a LMHeadModel model."
        )
