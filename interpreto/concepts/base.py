from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from interpreto.attributions.base import AttributionExplainer
from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.typing import ConceptsActivations, LatentActivation, ModelInput


class AbstractConceptExplainer(ABC):
    """
    Abstract class defining an interface for concept explanation.

    Attributes:
        splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
    """

    _differentiable_concept_encoder = False

    def __init__(self, splitted_model: ModelSplitterPlaceholder):
        """
        Initializes the concept explainer with a given model.

        Args:
            splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
        """
        self.splitted_model = splitted_model

    @abstractmethod
    def fit(self, inputs, split):
        """
        Abstract method for explaining a concept.

        Args:
            activations (LatentActivation): the activations to train the concept encoder on.
            split: The dataset split to use for training the concept encoder.

        Returns:
            The concept encoder.
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

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        print(inputs, concept, k)
        raise NotImplementedError()

    def top_k_inputs_for_concept_from_activations(
        self,
        inputs: ModelInput,
        activations: LatentActivation,
        corresponding_inputs: list[str],
        concept: int | list[int],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most important tokens/words/clauses/phrases related to a given concept.

        Args:
            inputs: The input data, which can be a string, a list of tokens/words/clauses/sentences, or a dataset.
            concept: The concept index (or list of concepts indices) to analyze.
            k: The number of important textual elements to retrieve. Defaults to 5.

        Returns:
            A list of tuples containing the top-k most relevant textual elements and their importance scores.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        print(inputs, concept, k)
        raise NotImplementedError()

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

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        print(inputs, concept, attribution_method)
        raise NotImplementedError()


class ConceptBottleneckExplainer(AbstractConceptExplainer):
    """
    Implementation of a concept explainer based on the Concept Bottleneck model.

    Attributes:
        splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
        concept_encoder_decoder (ConceptEncoderDecoder): Concept encoder-decoder
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        _differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    _differentiable_concept_decoder = False

    @abstractmethod
    def fit(self, activations: dict[LatentActivation], split: str):
        """
        Method for explaining a concept.

        Args:
            activations (LatentActivation): the activations to train the concept encoder-decoder on.
            split: The dataset split to use for training the concept encoder and decoder.

        Returns:
            Tuple containing the concept encoder and the concept decoder.
        """
        pass

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

    def encode_activations(self, activations: LatentActivation, **kwargs) -> ConceptsActivations:
        """
        Encode the given activations using the concept encoder-decoder.

        Args:
            activations (LatentActivation): The activations to encode.

        Returns:
            ConceptsActivations: The encoded activations.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        return self.concept_encoder_decoder.encode(activations, **kwargs)

    def decode_concepts(self, concepts: ConceptsActivations) -> LatentActivation:
        """
        Decode the given concepts using the concept encoder-decoder.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            LatentActivation: The decoded activations.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        return self.concept_encoder_decoder.decode(concepts)

    def get_dictionary(self) -> torch.Tensor:
        """
        Get the dictionary learned by the concept encoder-decoder.

        Returns:
            torch.Tensor: The learned dictionary.
        """
        assert self.fitted, "Concept explainer has not been fitted yet."
        return self.concept_encoder_decoder.get_dictionary()
