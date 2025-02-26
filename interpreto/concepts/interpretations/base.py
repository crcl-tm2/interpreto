from abc import ABC, abstractmethod


class ConceptInterpretability(ABC):
    """
    A collection of abstract interpretation methods for concepts.
    """

    @abstractmethod
    def interpret(self, concept: str):
        """
        Abstract method for interpreting a concept.
        This should be implemented in subclasses.

        :param concept: The concept to analyze.
        """
