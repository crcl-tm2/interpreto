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
Bases classes for concept explainers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from beartype import beartype
from beartype.typing import List
from jaxtyping import Float, Int, jaxtyped

from interpreto.attributions.base import AttributionExplainer
from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.typing import ConceptsActivations, LatentActivations, ModelInput


class AbstractConceptExplainer(ABC):
    """
    Abstract class defining an interface for concept explanation.

    Attributes:
        splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
        split (str): The split in the model where the concepts are encoded from.
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
    """

    _differentiable_concept_encoder: bool = False

    def __init__(self, splitted_model: ModelSplitterPlaceholder):
        """
        Initializes the concept explainer with a given model.

        Args:
            splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
        """
        if not isinstance(splitted_model, ModelSplitterPlaceholder):
            raise TypeError("Model should be a ModelSplitterPlaceholder.")
        self.splitted_model: ModelSplitterPlaceholder = splitted_model

    def verify_activations(
        self, activations: dict[str, LatentActivations] | LatentActivations, split: str | None = None
    ) -> tuple(LatentActivations, str | None):
        """
        Verify that the given activations are valid for the concept explainer.
        That is if the split corresponds to the model splits.

        Args:
            activations (dict[str, LatentActivations]): the activations to verify.
            split: (str | None): The dataset split to use for training the concept encoder. If None, the model is assumed to be a single-split model. And split is inferred from the keys of the activations dict.
        """
        if hasattr(self, "split") and self.split is not None:
            split: str = self.split

        if isinstance(activations, dict):
            if len(activations) != len(self.splitted_model.splits):
                raise ValueError(
                    f"Activations should be a dict with {self.splitted_model.splits} keys but got {activations.keys()}."
                )
            if split is None:
                if len(activations) != 1 and len(self.splitted_model.splits) != 1:
                    raise ValueError("Cannot infer split if the model is a not a single-split model.")
                split: str = list(activations.keys())[0]
        elif split is None:
            if len(self.splitted_model.splits) != 1:
                raise ValueError("Cannot infer split if the model is a not a single-split model.")
            split: str = self.splitted_model.splits[0]
            # assuming the users knows what they are doing
            return activations, None

        assert split in activations, f"Split {split} not found in activations."
        assert split in self.splitted_model.splits, f"Split {split} not found in model."

        return activations[split], split

    @abstractmethod
    def fit(self, activations: dict[str, LatentActivations], split: str | None = None):
        """
        Defines the concept encoder, thus the concept space, using the given activations.

        Args:
            activations (dict[str, LatentActivations]): the activations to train the concept encoder on.
            split: (str | None): The dataset split to use for training the concept encoder. If None, the model is assumed to be a single-split model. And split is inferred from the keys of the activations dict.

        Returns:
            The concept encoder.
        """
        pass

    @abstractmethod
    def encode_activations(
        self, activations: LatentActivations | dict[str, LatentActivations], **kwargs
    ) -> ConceptsActivations:
        """
        Encode the given activations using the concept encoder-decoder.

        Args:
            activations (LatentActivations | dict[str, LatentActivations]): The activations to encode.

        Returns:
            ConceptsActivations: The encoded activations.
        """
        pass

    def top_k_inputs_for_concept(
        self, inputs: ModelInput, concept: int | list[int], k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most important tokens/words/clauses/phrases related to a given concept.

        Args:
            inputs: The input data, which can be a string, a list of tokens/words/clauses/sentences, or a dataset.
            concept: The concept index (or list of concepts indices) to analyze.
            k: The number of important textual elements to retrieve. Defaults to 5.

        Returns:
            A list of tuples containing the top-k most relevant textual elements and their importance scores.
        """
        print(inputs, concept, k)
        raise NotImplementedError()

    @jaxtyped
    @beartype
    def top_k_tokens_for_concept_from_activations(
        self,
        inputs: List[List[str]],  # ModelInput
        activations: Float[torch.Tensor, "n l d"],  # LatentActivations
        concepts_indices: int | list[int] | Literal["all"] = "all",
        k: int = 5,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Retrieves the top-k most important tokens/words/clauses/phrases related to a given concept.

        Args:
            inputs (List[List[str]]): TODO: clarify the supported types of inputs
            activations (Float[torch.Tensor, "n l d"]): The activations to use for the analysis. They should correspond to the inputs. Shape: (n, l, d)
            concepts_indices (int | list[int] | Literal["all"]): The concept index (or list of concepts indices) to analyze. Defaults to "all".
            k (int): The number of important textual elements to retrieve. Defaults to 5.

        Returns:
            interpretation_dict (dict[int, list[tuple[str, float]]]): A dictionary with keys corresponding to the `concept` index and values containing the top-k most relevant textual elements and their importance scores.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        assert len(inputs) == len(activations), "Number of inputs and activations should be the same."
        n_tokens = len(inputs[0])

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
        topk_activations: Float[torch.Tensor, "cpt_of_interest k"] = topk_output[0].T
        topk_indices: Int[torch.Tensor, "cpt_of_interest k"] = topk_output[1].T

        interpretation_dict = {}
        # iterate over required concepts
        for c, top_indices, top_activations in zip(concepts_indices, topk_indices, topk_activations, strict=True):
            interpretation_dict[c] = [
                (
                    inputs[top_indices[rank] // n_tokens][top_indices[rank] % n_tokens],  # the token
                    top_activations[rank].item(),  # the corresponding concept activation
                )
                for rank in range(k)
            ]

        return interpretation_dict

    def input_concept_attribution(
        self,
        inputs: ModelInput,
        concept: int | list[int],
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """
        Computes the attribution of each input to a given concept.

        Args:
            inputs: The input data, which can be a string, a list of tokens/words/clauses/sentences, or a dataset.
            concept: The concept index (or list of concepts indices) to analyze.
            attribution_method: The method to use for attribution analysis.

        Returns:
            A list of attribution scores for each input.
        """
        print(inputs, concept, attribution_method)
        raise NotImplementedError()


class ConceptBottleneckExplainer(AbstractConceptExplainer):
    """
    The concept bottleneck explainer can encode activations into concepts and decode concepts into activations.

    Attributes:
        splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
        split (str): The split in the model where the concepts are encoded from.
        concept_encoder_decoder (ConceptEncoderDecoder): Concept encoder-decoder
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        _differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    _differentiable_concept_decoder: bool = False

    def encode_activations(
        self, activations: LatentActivations | dict[str, LatentActivations], **kwargs
    ) -> ConceptsActivations:
        """
        Encode the given activations using the concept encoder-decoder.

        Args:
            activations (LatentActivations | dict[str, LatentActivations]): The activations to encode.

        Returns:
            ConceptsActivations: The encoded activations.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."

        inputs: LatentActivations
        inputs, _ = self.verify_activations(activations)
        inputs = inputs.to(self.concept_encoder_decoder.device)

        return self.concept_encoder_decoder.encode(inputs, **kwargs)

    def decode_concepts(self, concepts: ConceptsActivations) -> LatentActivations:
        """
        Decode the given concepts using the concept encoder-decoder.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            LatentActivations: The decoded activations.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        concepts = concepts.to(self.concept_encoder_decoder.device)
        return self.concept_encoder_decoder.decode(concepts)

    def get_dictionary(self) -> torch.Tensor:
        """
        Get the dictionary learned by the concept encoder-decoder.

        Returns:
            torch.Tensor: The learned dictionary.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        return self.concept_encoder_decoder.get_dictionary()

    def concept_output_attribution(
        self, concepts: ConceptsActivations, attribution_method: type[AttributionExplainer], **attribution_kwargs
    ):
        """
        Computes the attribution of each concept to a given example.

        Args:
            concepts: The concepts to analyze.
            attribution_method: The method to use for attribution analysis.

        Returns:
            A list of attribution scores for each concept.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        raise NotImplementedError(
            f"Concept output attribution method {concepts}, {attribution_method} is not implemented yet."
        )
