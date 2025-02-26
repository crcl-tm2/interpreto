"""
Base classes for perturbations used in attribution methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

from interpreto.typing import ModelInput, TokenEmbedding


class Perturbator(ABC):
    """
    Object allowing you to perturbate an input (add noise, change tokens, create progression of vectors...)
    """

    @abstractmethod
    def perturbate(self, item: ModelInput | Collection[ModelInput]) -> Collection[TokenEmbedding]:
        """
        Method to perturbate an input, should return a collection of perturbated elements and their associated masks
        """
        perturbated_elements = ...
        masks = ...
        return perturbated_elements, masks


class TokenPerturbation(Perturbator):
    """
    Generic class for token modification (occlusion, words substitution...)
    """


class TensorPerturbation(Perturbator):
    """
    Generic class for any tensorwise modification
    """
