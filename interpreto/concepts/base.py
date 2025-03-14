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
Bases Classes for Concept-based Explainers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from textwrap import dedent
from typing import Any, Generic, TypeVar

import torch
from overcomplete.base import BaseDictionaryLearning
from torch import nn
from transformers import BatchEncoding

from interpreto.attributions.base import AttributionExplainer
from interpreto.commons.model_wrapping.model_with_split_points import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.typing import ConceptsActivations, LatentActivations, ModelInput

Module = TypeVar("Module", bound=nn.Module)
T = TypeVar("T")


# Decorator that checks if the concept model is fitted before calling the method
def check_fitted(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(self: AbstractConceptExplainer, *args, **kwargs) -> T:
        if not self.is_fitted or self.split_point is None:
            raise RuntimeError("Concept encoder is not fitted yet. Use the .fit() method to fit the explainer.")
        return func(self, *args, **kwargs)

    return wrapper


class AbstractConceptExplainer(ABC, Generic[Module]):
    """Abstract class defining an interface for concept explanation.
    Child classes should implement the `fit` and `encode_activations` methods, and only assume the presence of an
        encoding step using the `concept_model` to convert activations to latent concepts.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str): The split point used to train the `concept_model`.
        concept_model (torch.nn.Module): The model used to extract concepts from the activations of
            `model_with_split_points`. The only assumption for classes inheriting from this class is that
            the `concept_model` can encode activations into concepts with `encode_activations`.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: Module,
        split_point: str | None = None,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model (torch.nn.Module): The model used to extract concepts from
                the activations of `model_with_split_points`.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
        """
        if not isinstance(model_with_split_points, ModelWithSplitPoints):
            raise TypeError(
                f"The given model should be a ModelWithSplitPoints, but {type(model_with_split_points)} was given."
            )
        self.model_with_split_points: ModelWithSplitPoints = model_with_split_points
        self.concept_model: Module = concept_model
        self.split_point: str = self.get_and_verify_split_point(split_point)
        self.__is_fitted: bool = False
        self.has_differentiable_concept_encoder = False

    @property
    def is_fitted(self) -> bool:
        return self.__is_fitted

    def __repr__(self):
        return dedent(f"""\
            {self.__class__.__name__}(
                split_point={self.split_point},
                concept_model={type(self.concept_model).__name__},
                is_fitted={self.is_fitted},
                has_differentiable_concept_encoder={self.has_differentiable_concept_encoder},
            )""")

    @abstractmethod
    def fit(self, activations: LatentActivations | dict[str, LatentActivations], *args, **kwargs) -> Any:
        """Fits `concept_model` on the given activations.

        Args:
            activations (dict[str, torch.Tensor]): A dictionary with model paths as keys and the corresponding
                tensors as values.

        Returns:
            `None`, `concept_model` is fitted in-place, `is_fitted` is set to `True` and `split_point` is set.
        """
        pass

    @abstractmethod
    def encode_activations(self, activations: LatentActivations) -> ConceptsActivations:
        """Abstract method defining how activations are converted into concepts by the concept encoder.

        Args:
            activations (torch.Tensor): The activations to encode.

        Returns:
            A `torch.Tensor` of encoded activations produced by the fitted concept encoder.
        """
        pass

    def verify_activations(self, activations: dict[str, LatentActivations]) -> None:
        """
        Verify that the given activations are valid for the `model_with_split_points` and `self.split_point`.
        Cases in which the activations are not valid include:

        * Activations are not a valid dictionary.
        * Specified split point does not exist in the activations.

        Args:
            activations (dict[str, torch.Tensor]): A dictionary with model paths as keys and the corresponding
                tensors as values.
        """
        if not isinstance(activations, dict) or not all(isinstance(act, torch.Tensor) for act in activations.values()):
            raise TypeError(
                "Invalid activations for the concept explainer. "
                "Activations should be a dictionary of model paths and torch.Tensor activations. "
                f"Got: '{type(activations)}'"
            )
        activations_split_points = list(activations.keys())
        if self.split_point not in activations_split_points:
            raise ValueError(
                f"Fitted split point '{self.split_point}' not found in activations.\n"
                f"Available split_points: {', '.join(activations_split_points)}."
            )

    def get_and_verify_split_point(self, split_point: str | None) -> str:
        if split_point is None and len(self.model_with_split_points.split_points) > 1:
            raise ValueError(
                "If the model has more than one split point, a split point for fitting the concept model should "
                f"be specified. Got split point: '{split_point}' with model split points: "
                f"{', '.join(self.model_with_split_points.split_points)}."
            )
        if split_point is None:
            split_point = self.model_with_split_points.split_points[0]
        if split_point not in self.model_with_split_points.split_points:
            raise ValueError(
                f"Split point '{split_point}' not found in model split points: "
                f"{', '.join(self.model_with_split_points.split_points)}."
            )
        return split_point

    def prepare_fit(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
        overwrite: bool,
    ) -> LatentActivations:
        if self.is_fitted and not overwrite:
            raise RuntimeError(
                "Concept explainer has already been fitted. Refitting will overwrite the current model."
                "If this is intended, use `overwrite=True` in fit(...)."
            )
        if isinstance(activations, dict):
            self.verify_activations(activations)
            split_activations = activations[self.split_point]
        else:
            split_activations = activations
        assert len(split_activations.shape) == 2, (
            f"Input activations should be a 2D tensor of shape (batch_size, n_features) but got {split_activations.shape}."
        )
        return split_activations

    @check_fitted
    def top_k_inputs_for_concept(
        self,
        inputs: str | list[str] | BatchEncoding,
        concepts_indices: int | list[int] | Literal[all] = "all",
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most important input elements (tokens/words/clauses/phrases) related to a given concept.

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            concepts_indices (int | list[int] | Literal["all"]): The concept index (or list of concepts indices) to analyze. Defaults to "all".
            k: The number of input elements to retrieve. Defaults to 5.

        Returns:
            An ordered list of k tuples, each containing containing one of the top-k relevant input elements
                for the specified concept and its importance scores.
        """
        self.split_point: str  # Verified by check_fitted
        activations = self.model_with_split_points.get_activations(
            inputs, select_strategy=ActivationSelectionStrategy.FLATTEN
        )
        splitted_inputs = [
            self.model_with_split_points.tokenizer.tokenize(input_sentence) for input_sentence in inputs
        ]  # TODO: see if we do not wrap this function in the model_with_split_points
        return self.top_k_inputs_for_concept_from_activations(
            splitted_inputs, activations[self.split_point], concepts_indices, k
        )

    @check_fitted
    def top_k_inputs_for_concept_from_activations(
        self,
        splitted_inputs: list[list[str]],
        activations: LatentActivations | dict[str, LatentActivations],
        concepts_indices: int | list[int] | Literal[all] = "all",
        k: int = 5,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Retrieves the top-k most important input elements (tokens/words/clauses/phrases) related to a given concept.
        This version of the method uses pre-computed activations

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            activations (LatentActivations | dict[str, LatentActivations]): The activations to use for the analysis. They should correspond to the inputs. Shape: (n*l, d)
            concepts_indices (int | list[int] | Literal["all"]): The concept index (or list of concepts indices) to analyze. Defaults to "all".
            k (int): The number of important textual elements to retrieve. Defaults to 5.

        Returns:
            interpretation_dict (dict[int, list[tuple[str, float]]]): A dictionary with keys corresponding to the `concept` index and values containing the top-k most relevant textual elements and their importance scores.
        """
        if isinstance(activations, dict):
            self.verify_activations(activations)
            split_activations = activations[self.split_point]
        else:
            split_activations = activations
        if not len(split_activations) % len(splitted_inputs) == 0:
            raise ValueError(
                f"Number of inputs and activations should be the same. Got {len(splitted_inputs)} inputs and {len(split_activations)} activations."
            )
        n_tokens = len(split_activations) // len(splitted_inputs)

        # Shape: (n*l, d)
        flattened_activations: LatentActivations = activations.view(-1, activations.shape[-1])

        # Shape: (n*l, cpt)
        concepts_activations: ConceptsActivations = self.encode_activations(flattened_activations)

        # take subset of concepts as specified by the user
        if concepts_indices != "all":
            if isinstance(concepts_indices, int):
                concepts_indices = [concepts_indices]

            if not isinstance(concepts_indices, list) and not all(isinstance(c, int) for c in concepts_indices):
                raise ValueError(
                    f"`concepts_indices` should be 'all', an int, or a list of int. Received {concepts_indices}."
                )

            if max(concepts_indices) >= concepts_activations.shape[1]:
                raise ValueError(
                    f"At least one concept index out of bounds. `max(concepts_indices)`: {max(concepts_indices)} >= {concepts_activations.shape[1]}."
                )

            # Shape: (n*l, cpt_of_interest)
            concepts_activations: ConceptsActivations = concepts_activations.T[concepts_indices].T
        else:
            concepts_indices = list(range(concepts_activations.shape[1]))

        # extract indices of the top-k input tokens for each specified concept
        topk_output = torch.topk(concepts_activations, k=k, dim=0)
        topk_activations = topk_output[0].T  # Shape: (cpt_of_interest, k)
        topk_indices = topk_output[1].T  # Shape: (cpt_of_interest, k)

        interpretation_dict = {}
        # iterate over required concepts
        for c, top_indices, top_activations in zip(concepts_indices, topk_indices, topk_activations, strict=True):
            interpretation_dict[c] = [
                (
                    splitted_inputs[top_indices[rank] // n_tokens][top_indices[rank] % n_tokens],  # the token
                    top_activations[rank].item(),  # the corresponding concept activation
                )
                if top_indices[rank] % n_tokens < len(splitted_inputs[top_indices[rank] // n_tokens])
                else "[PAD]"
                for rank in range(k)
            ]

        # TODO: find a way to remove duplicate in the topk inputs but keep k outputs

        return interpretation_dict

    @check_fitted
    def input_concept_attribution(
        self,
        inputs: ModelInput,
        concept: int,
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """Attributes model inputs for a selected concept.

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            concept (int): Index identifying the position of the concept of interest (score in the
                `ConceptActivations` tensor) for which relevant input elements should be retrieved.
            attribution_method: The attribution method to obtain importance scores for input elements.

        Returns:
            A list of attribution scores for each input.
        """
        raise NotImplementedError("Input-to-concept attribution method is not implemented yet.")


class ConceptBottleneckExplainer(AbstractConceptExplainer[BaseDictionaryLearning], ABC):
    """A concept bottleneck explainer wraps a `concept_model` that should be able to encode activations into concepts
    and decode concepts into activations.

    We use the term "concept bottleneck" loosely, as the latent space can be overcomplete compared to activation
        space, as in the case of sparse autoencoders.

    We assume that the concept model follows the structure of an [`overcomplete.BaseDictionaryLearning`](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)
    model, which defines the `encode` and `decode` methods for encoding and decoding activations into concepts.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str): The split point used to train the `concept_model`.
        concept_model ([BaseDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)): The model used to extract concepts from the
            activations of  `model_with_split_points`. The only assumption for classes inheriting from this class is
            that the `concept_model` can encode activations into concepts with `encode_activations`.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: BaseDictionaryLearning,
        split_point: str | None,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model ([BaseDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)): The model used to extract concepts from
                the activations of `model_with_split_points`.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
        """
        super().__init__(model_with_split_points, concept_model, split_point)
        self.has_differentiable_concept_decoder = False

    @property
    def is_fitted(self) -> bool:
        return self.concept_model.fitted

    def __repr__(self):
        return dedent(f"""\
            {self.__class__.__name__}(
                split_point={self.split_point},
                concept_model={type(self.concept_model).__name__},
                is_fitted={self.is_fitted},
                has_differentiable_concept_encoder={self.has_differentiable_concept_encoder},
                has_differentiable_concept_decoder={self.has_differentiable_concept_decoder},
            )""")

    @check_fitted
    def encode_activations(self, activations: LatentActivations) -> torch.Tensor:  # ConceptActivations
        """Encode the given activations using the `concept_model` encoder.

        Args:
            activations (LatentActivations): The activations to encode.

        Returns:
            The encoded concept activations.
        """
        self.verify_activations({self.split_point: activations})
        return self.concept_model.encode(activations)  # type: ignore

    @check_fitted
    def decode_concepts(self, concepts: ConceptsActivations) -> torch.Tensor:  # LatentActivations
        """Decode the given concepts using the `concept_model` decoder.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            The decoded model activations.
        """
        return self.concept_model.decode(concepts)  # type: ignore

    @check_fitted
    def get_dictionary(self) -> torch.Tensor:  # TODO: add this to tests
        """Get the dictionary learned by the fitted `concept_model`.

        Returns:
            torch.Tensor: A `torch.Tensor` containing the learned dictionary.
        """
        return self.concept_model.get_dictionary()  # type: ignore

    @check_fitted
    def concept_output_attribution(
        self,
        inputs: ModelInput,
        concepts: ConceptsActivations,
        target: int,
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """Computes the attribution of each concept for the logit of a target output element.

        Args:
            inputs (ModelInput): An input datapoint for the model.
            concepts (torch.Tensor): Concept activation tensor.
            target (int): The target class for which the concept output attribution should be computed.
            attribution_method: The attribution method to obtain importance scores for input elements.

        Returns:
            A list of attribution scores for each concept.
        """
        raise NotImplementedError("Concept-to-output attribution method is not implemented yet.")
