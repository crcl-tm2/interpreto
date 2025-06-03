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
Base class for concept interpretation methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from jaxtyping import Float
from nnsight.intervention.graph import InterventionProxy

from interpreto.commons import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations, ModelInput


class BaseConceptInterpretationMethod(ABC):
    """Code: [:octicons-mark-github-24: `concepts/interpretations/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/interpretations/base.py)

    Abstract class defining an interface for concept interpretation.
    Its goal is to make the dimensions of the concept space interpretable by humans.

    Attributes:
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        split_point: str,
        concept_model: ConceptModelProtocol,
    ):
        if not hasattr(concept_model, "encode"):
            raise TypeError(
                f"Concept model should be able to encode activations into concepts. Got: {type(concept_model)}."
            )

        if split_point not in model_with_split_points.split_points:
            raise ValueError(
                f"Split point '{split_point}' not found in model split points: "
                f"{', '.join(model_with_split_points.split_points)}."
            )

        self.model_with_split_points: ModelWithSplitPoints = model_with_split_points
        self.split_point: str = split_point
        self.concept_model: ConceptModelProtocol = concept_model

    @abstractmethod
    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: ModelInput | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
        use_vocab: bool = False,
    ) -> Mapping[int, Any]:
        """
        Interpret the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and an object allowing to interpret them.
        It can be a label, a description, examples, etc.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (ModelInput | None): The inputs to use for the interpretation.
            latent_activations (LatentActivations | None): The latent activations to use for the interpretation.
            concepts_activations (ConceptsActivations | None): The concepts activations to use for the interpretation.
            use_vocab (bool): Whether to use the vocabulary for the interpretation.

        Returns:
            Mapping[int, Any]: The interpretation of each of the specified concepts.
        """
        raise NotImplementedError

    def concepts_activations_from_source(
        self,
        *,
        inputs: list[str] | None = None,
        latent_activations: Float[torch.Tensor, "nl d"] | None = None,
        concepts_activations: Float[torch.Tensor, "nl cpt"] | None = None,
        use_vocab: bool = False,
    ) -> tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
        if use_vocab:
            return self._concepts_activations_from_vocab()

        if inputs is None:
            raise ValueError("Inputs are required to compute concept activations.")

        if concepts_activations is not None:
            return inputs, concepts_activations

        if latent_activations is not None:
            concepts_activations = concept_model.encode(latent_activations)
            return inputs, concepts_activations

        activations_dict: InterventionProxy = self.model_with_split_points.get_activations(
            inputs, select_strategy=ActivationSelectionStrategy.FLATTEN
        )
        latent_activations = self.model_with_split_points.get_split_activations(
            activations_dict, split_point=self.split_point
        )
        concepts_activations = self.concept_model.encode(latent_activations)
        return inputs, concepts_activations

    def _concepts_activations_from_vocab(
        self,
    ) -> tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
        """
        Computes the concepts activations for each token of the vocabulary

        Args:
            model_with_split_points (ModelWithSplitPoints):
            split_point (str):
            concept_model (ConceptModelProtocol):

        Returns:
            tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
                - The list of tokens in the vocabulary
                - The concept activations for each token
        """
        # extract and sort the vocabulary
        vocab_dict: dict[str, int] = self.model_with_split_points.tokenizer.get_vocab()
        input_ids: list[int]
        inputs, input_ids = zip(*vocab_dict.items(), strict=True)  # type: ignore

        input_tensor: Float[torch.Tensor, "v 1"] = torch.tensor(input_ids).unsqueeze(1)
        activations_dict: InterventionProxy = self.model_with_split_points.get_activations(
            input_tensor, select_strategy=ActivationSelectionStrategy.FLATTEN
        )  # TODO: verify `ModelWithSplitPoints.get_activations()` can take in ids
        latent_activations = self.model_with_split_points.get_split_activations(
            activations_dict, split_point=split_point
        )
        concepts_activations = self.concept_model.encode(latent_activations)
        return inputs, concepts_activations  # type: ignore


def verify_concepts_indices(
    concepts_activations: ConceptsActivations,
    concepts_indices: int | list[int],
) -> list[int]:
    # take subset of concepts as specified by the user
    if isinstance(concepts_indices, int):
        concepts_indices = [concepts_indices]

    if not isinstance(concepts_indices, list) or not all(isinstance(c, int) for c in concepts_indices):
        raise ValueError(f"`concepts_indices` should be 'all', an int, or a list of int. Received {concepts_indices}.")

    if max(concepts_indices) >= concepts_activations.shape[1] or min(concepts_indices) < 0:
        raise ValueError(
            f"At least one concept index out of bounds. `max(concepts_indices)`: {max(concepts_indices)} >= {concepts_activations.shape[1]}."
        )

    return concepts_indices
