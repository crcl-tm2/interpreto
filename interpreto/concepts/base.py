from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class AbstractConceptExplainer(ABC):
    """
    Abstract class defining an interface for concept explanation.
    """

    def __init__(self, model, concept_extraction_method):
        """
        Initializes the concept explainer with a given model.

        :param model: The model to apply the explanation on.
        """
        self.model = model
        self.concept_extraction_method = concept_extraction_method

    @abstractmethod
    def fit(self, inputs, split):
        """
        Abstract method for explaining a concept.

        :param inputs: The inputs to be explained.
        :return: the concept encoder.
        """
        # self.encoder = ...
        # self.split = split

    def top_k_inputs_for_concept(
        self, inputs: str | Iterable[str] | Iterable[Iterable[str]], concept: str | list[str], k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most important tokens/words/clauses/phrases related to a given concept.

        :param inputs: The input data (can be a string, a list of tokens/words/clauses/sentences, or a dataset).
        :param concept: The concept (or list of concepts) to analyze.
        :param k: The number of important textual elements to retrieve.
        :return: A list of tuples containing the top-k most relevant textual elements and their importance scores.
        """
        # TODO
        print(inputs, concept, k)
        raise NotImplementedError()

    def input_concept_attribution(
        self, inputs: str | Iterable[str] | Iterable[Iterable[str]], concept: str | list[str], attribution_method: str
    ) -> list[float]:
        """
        Computes the attribution of each input to a given concept.

        :param inputs: The input data (can be a string, a list of tokens/words/clauses/sentences, or a dataset).
        :param concept: The concept (or list of concepts) to analyze.
        :param attribution_method: The method to use for attribution analysis.
        :return: A list of attribution scores for each input.
        """
        # TODO
        print(inputs, concept, attribution_method)
        raise NotImplementedError()
