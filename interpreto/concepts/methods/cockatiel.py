"""
Implementation of the Cockatiel concept explainer TODO: add paper
"""

from __future__ import annotations

from interpreto.attributions import Occlusion, SobolAttribution
from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.methods.overcomplete_cbe import OvercompleteDictionaryLearning, OvercompleteMethods
from interpreto.typing import ConceptsActivations, ModelInput


class Cockatiel(OvercompleteDictionaryLearning):
    """
    Implementation of the Cockatiel concept explainer

    # TODO: add paper

    Attributes:
        splitted_model (ModelSplitterPlaceholder): Model splitter
        concept_encoder_decoder (oc_opt.NMF): Overcomplete NMF concept encoder decoder
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        _differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    def __init__(self, splitted_model: ModelSplitterPlaceholder, n_concepts: int, device: str = "cpu"):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete concept-encoder-decoder framework.

        Args:
            splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
            n_concepts (int): Number of concepts to explain.
            device (str): Device to use for the concept encoder-decoder.
        """
        super().__init__(
            splitted_model=splitted_model,
            ConceptEncoderDecoder=OvercompleteMethods.NMF,
            n_concepts=n_concepts,
            device=device,
        )

    def input_concept_attribution(
        self, inputs: ModelInput, concept: int | list[int], **attribution_kwargs
    ) -> list[float]:
        """
        Computes the attribution of each input to a given concept.

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences, or a dataset.
            concept (int | list[int]): The concept index (or list of concepts indices) to analyze.

        Returns:
            A list of attribution scores for each input.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        return super().input_concept_attribution(inputs, concept, Occlusion, **attribution_kwargs)

    def concept_output_attribution(self, concepts: ConceptsActivations, **attribution_kwargs):
        """
        Computes the attribution of each concept to a given example.

        Args:
            concepts (ConceptsActivations): The concepts to analyze.

        Returns:
            A list of attribution scores for each concept.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        super().concept_output_attribution(concepts, attribution_method=SobolAttribution, **attribution_kwargs)
