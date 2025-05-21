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
from collections.abc import Callable, Iterable, MutableMapping
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
from interpreto.commons.model_wrapping.inference_wrapper import InferenceModes, InferenceWrapper
from interpreto.typing import ClassificationTarget, GeneratedTarget, ModelInputs, TensorMapping

SingleAttribution = (
    Float[torch.Tensor, "l"] | Float[torch.Tensor, "l c"] | Float[torch.Tensor, "l l_g"] | Float[torch.Tensor, "l l_t"]
)


def clone_tensor_mapping(tm: TensorMapping, detach: bool = False) -> TensorMapping:
    """
    Clone a TensorMapping, optionally detaching the tensors.

    Args:
        tm (TensorMapping): tensor mapping to clone
        detach (bool, optional): specify if new tensors must be detached. Defaults to False.

    Returns:
        TensorMapping: cloned tensor mapping
    """
    return {k: v.detach().clone() if detach else v.clone() for k, v in tm.items()}


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
                    - Classification (single class): `(l)`
                    - Classification (all classes): `(c, l)`, where `c` is the number of classes.
                    - Generative models: `(l_g, l)`, where `l_g` is the length of the generated part.
                        - For non-generated elements, there are `l_g` attribution scores.
                        - For generated elements, scores are zero for previously generated tokens.
                    - Token classification: `(l_t, l)`, where `l_t` is the number of token classes. When the tokens are disturbed, l = l_t.
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
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity_level: GranularityLevel = GranularityLevel.DEFAULT,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,  # TODO: add to all classes
    ) -> None:
        """
        Initializes the AttributionExplainer.

        Args:
            model (PreTrainedModel): The model to be explained.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
            batch_size (int): The batch size used for model inference.
            perturbator (Perturbator, optional): Instance used to generate input perturbations.
                If None, the perturbator returns only the original input.
            aggregator (Aggregator, optional): Instance used to aggregate computed attribution scores.
                If None, the aggregator returns the original scores.
            device (torch.device, optional): The device on which computations are performed.
                If None, defaults to the device of the model.
            granularity_level (GranularityLevel, optional): The level of granularity for the explanation (e.g., token, word, sentence).
                Defaults to GranularityLevel.DEFAULT (TOKEN)
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
        """
        self.tokenizer = tokenizer
        self.inference_wrapper = self._associated_inference_wrapper(
            model, batch_size=batch_size, device=device, mode=inference_mode
        )  # type: ignore
        self.perturbator = perturbator or Perturbator()
        self.perturbator.to(self.device)
        self.aggregator = aggregator or Aggregator()
        self.granularity_level = granularity_level

        # TODO : check these line, eventually move them
        # give pad token id to the inference wrapper
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_wrapper.pad_token_id = self.tokenizer.pad_token_id  # type: ignore

    def get_scores(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: Iterable[torch.Tensor],
    ) -> Iterable[torch.Tensor]:
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
            model_inputs (str, TensorMapping, or Iterable): The raw model inputs.

        Returns:
            List[TensorMapping]: A list of processed model input mappings.

        Raises:
            ValueError: If the type of model_inputs is not supported.
        """
        if isinstance(model_inputs, str):
            return [
                self.tokenizer(
                    model_inputs, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True
                )
            ]
        if isinstance(
            model_inputs, MutableMapping
        ):  # we cant use TensorMapping in the isinstance so we use MutableMapping.
            return [
                {key: value[i].unsqueeze(0) for key, value in model_inputs.items()}
                for i in range(model_inputs["attention_mask"].shape[0])  # type: ignore
            ]  # type: ignore
        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))
        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    @abstractmethod
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: torch.Tensor | Iterable[torch.Tensor] | None = None,
        **model_kwargs: Any,
    ) -> tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]:
        """
        Processes the inputs and targets for explanation.

        This method must be implemented by subclasses.

        Args:
            model_inputs (Iterable[TensorMapping]): The inputs to the model.
            targets (Any): The targets to be explained.
            model_kwargs (Any): Additional model-specific arguments.

        Returns:
            tuple: A tuple of (processed_inputs, processed_targets).

        Raises:
            NotImplementedError: Always raised. Subclasses must implement this method.
        """
        raise NotImplementedError(
            "Specific task subclasses must implement the 'process_inputs_to_explain_and_targets' method "
            "to correctly process inputs and targets for explanations."
        )

    def explain(
        self,
        model_inputs: ModelInputs,
        targets: torch.Tensor | Iterable[torch.Tensor] | None = None,
        **model_kwargs: Any,
    ) -> Iterable[AttributionOutput]:
        """
        Computes attributions for NLP models.

        Process:
            1. Process and standardize the model inputs.
            2. Create the tokenizer's pad token if not already set and add it to the inference wrapper.
            3. If targets are not provided, create them. Otherwise, for each input-target pair, process them.
            4. Decompose the inputs based on the desired granularity and decode tokens.
            5. Generate perturbations for the constructed inputs.
            6. Compute scores using either gradients (if use_gradient is True) or targeted logits.
            7. Aggregate the scores to obtain contribution values.

        Args:
            model_inputs (ModelInputs): Raw inputs for the model.
            targets (torch.Tensor | Iterable[torch.Tensor] | None): Targets for which explanations are desired.
            Further types might be supported by sub-classes.
            It depends on the task:
                - For classification tasks, encodes the target class or classes to explain.
                - For generation tasks, encodes the target text or tokens to explain.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        # Ensure the model inputs are in the correct format
        sanitized_model_inputs: Iterable[TensorMapping] = self.process_model_inputs(model_inputs)

        # Process the inputs and targets for explanation
        # If targets are not provided, create them from model_inputs_to_explain.
        model_inputs_to_explain: Iterable[TensorMapping]
        sanitized_targets: Iterable[torch.Tensor]
        model_inputs_to_explain, sanitized_targets = self.process_inputs_to_explain_and_targets(
            sanitized_model_inputs, targets, **model_kwargs
        )

        # Decompose each input for the desired granularity level (tokens, words, sentences...)
        decompositions = [
            GranularityLevel.get_decomposition(t, self.granularity_level) for t in model_inputs_to_explain
        ]

        # Create perturbation masks and perturb inputs based on the masks.
        # Inputs might be embedded during the perturbation process if the perturbator works with embeddings.
        pert_generator: Iterable[TensorMapping]
        mask_generator: Iterable[torch.Tensor | None]
        pert_generator, mask_generator = split_iterator(self.perturbator.perturb(m) for m in model_inputs_to_explain)

        # Compute the score on perturbed inputs:
        # - If use_gradient is True, compute gradients.
        # - Otherwise, compute targeted logits.
        scores: Iterable[torch.Tensor] = self.get_scores(
            pert_generator, (a.to(self.device) for a in sanitized_targets)
        )

        # Aggregate the scores using the aggregator function and the perturbation masks.
        contributions = (
            self.aggregator(score.detach(), mask.to(self.device) if mask is not None else None).squeeze(0)
            for score, mask in zip(scores, mask_generator, strict=True)
        )

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

    def __call__(self, model_inputs: ModelInputs, targets=None) -> Iterable[AttributionOutput]:
        """
        Enables the explainer instance to be called as a function.

        Args:
            model_inputs (ModelInputs): Raw inputs for the model.
            targets: Targets for which explanations are desired. It depends on the task:
                - For classification tasks, encodes the target class or classes to explain.
                - For generation tasks, encodes the target text or tokens to explain.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        return self.explain(model_inputs, targets)


class ClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for classification models
    """

    _associated_inference_wrapper = ClassificationInferenceWrapper

    def process_targets(self, targets: ClassificationTarget, batch_size: int = 1) -> Iterable[torch.Tensor]:
        if isinstance(targets, int):
            targets = torch.tensor([targets])
        if isinstance(targets, torch.Tensor):
            return targets.view(batch_size, 1, -1).split(1, dim=0)
        if isinstance(targets, Iterable):
            if all(isinstance(t, int) for t in targets):
                return self.process_targets(torch.tensor(targets), batch_size)
            return list(itertools.chain(*[self.process_targets(item, batch_size) for item in targets]))

    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: ClassificationTarget | None = None,
        **model_kwargs: Any,
    ) -> tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]:
        logits = torch.stack(
            [a.detach() for a in self.inference_wrapper.get_logits(clone_tensor_mapping(a) for a in model_inputs)]
        )
        if targets is None:
            targets = logits.argmax(dim=-1)
        # TODO : change call to process_target
        sanitized_targets: Iterable[torch.Tensor] = self.process_targets(targets, logits.shape[0])
        return model_inputs, sanitized_targets


class GenerationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for generation models
    """

    _associated_inference_wrapper = GenerationInferenceWrapper
    inference_wrapper: GenerationInferenceWrapper

    def process_targets(self, targets: GeneratedTarget) -> list[torch.Tensor]:
        """
        Processes the target inputs for generative models into a standardized format.

        This function handles various input types for targets (string, TensorMapping, or Iterable)
        and converts them into a list of tensors containing token IDs.

        Args:
            targets (str, TensorMapping, torch.Tensor, or Iterable): The target texts or tokens.

        Returns:
            List[torch.Tensor]: A list of tensors representing the target token IDs.

        Raises:
            ValueError: If the target type is not supported.
        """
        if isinstance(targets, str):
            return [self.tokenizer(targets, return_tensors="pt")["input_ids"]]  # type: ignore
        if isinstance(targets, MutableMapping):  # TensorMapping cannot be used in isinstance
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
        self, model_inputs: ModelInputs, targets: GeneratedTarget | None = None, **model_kwargs: dict[str, Any]
    ) -> tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]:
        """
        Processes the inputs and targets for the generative model.
        If targets are not provided, create them with model_inputs_to_explain. Otherwise, for each input-target pair:
            a. Embed the input.
            b. Embed the target and concatenate with the input embeddings.
            c. Construct a new input mapping that includes both embeddings.
        Then, add offsets mapping and special tokens mask.

        Args:
            model_inputs (ModelInputs): The raw inputs for the generative model.
            targets (GeneratedTarget): The target texts or tokens for which explanations are desired.
            model_kwargs (dict): Additional arguments for the generation process.

        Returns:
            tuple: A tuple containing a list of processed model inputs and a list of processed targets.
        """
        sanitized_targets: list[torch.Tensor]
        if targets is None:
            model_inputs_to_explain_basic, sanitized_targets = (
                self.inference_wrapper.get_inputs_to_explain_and_targets(model_inputs, **model_kwargs)
            )
        else:
            sanitized_targets = self.process_targets(targets)
            model_inputs_to_explain_basic = []
            for model_input, target in zip(model_inputs, sanitized_targets, strict=True):
                model_inputs_to_explain_basic.append(
                    {
                        "input_ids": torch.cat([model_input["input_ids"], target], dim=1),  # type: ignore
                        "attention_mask": torch.cat([model_input["attention_mask"], torch.ones_like(target)], dim=1),  # type: ignore
                    }
                )
        # Add offsets mapping and special tokens mask:
        model_inputs_to_explain_text = [
            self.tokenizer.decode(elem["input_ids"][0]) for elem in model_inputs_to_explain_basic
        ]
        model_inputs_to_explain = [
            self.tokenizer(
                [model_inputs_to_explain_text],
                return_tensors="pt",
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            for model_inputs_to_explain_text in model_inputs_to_explain_text
        ]

        return model_inputs_to_explain, sanitized_targets


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
