from abc import ABC, abstractmethod
from typing import List, Tuple, Union

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
        #self.encoder = ...
        #self.split = split
        pass

    def top_k_inputs_for_concept(self, inputs: Union[str, List[str], List[List[str]]], concept: Union[str, List[str]], k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k most important tokens/words/clauses/phrases related to a given concept.

        :param inputs: The input data (can be a string, a list of tokens/words/clauses/sentences, or a dataset).
        :param concept: The concept (or list of concepts) to analyze.
        :param k: The number of important textual elements to retrieve.
        :return: A list of tuples containing the top-k most relevant textual elements and their importance scores.
        """
        #TODO
        pass

    def input_concept_attribution(self, inputs: Union[str, List[str], List[List[str]]], concept: Union[str, List[str]], attribution_method: str) -> List[float]:
        """
        Computes the attribution of each input to a given concept.

        :param inputs: The input data (can be a string, a list of tokens/words/clauses/sentences, or a dataset).
        :param concept: The concept (or list of concepts) to analyze.
        :param attribution_method: The method to use for attribution analysis.
        :return: A list of attribution scores for each input.
        """
        #TODO
        pass
