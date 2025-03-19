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

from collections import Counter
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

import torch
from jaxtyping import Float

from interpreto.commons import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.concepts.interpretations.base import BaseConceptInterpretationMethod
from interpreto.typing import ConceptModel, ConceptsActivations, LatentActivations


class Granularities(Enum):  # TODO: harmonize with attribution granularities
    # TODO: doc strings
    # ALL_TOKENS = "all_tokens"
    TOKENS = "tokens"
    # WORDS = "words"
    # CLAUSES = "clauses"
    # SENTENCES = "sentences"


class InterpretationSources(Enum):
    CONCEPTS_ACTIVATIONS = "concepts_activations"
    LATENT_ACTIVATIONS = "latent_activations"
    INPUTS = "inputs"
    VOCABULARY = "vocabulary"
    AUTO = "auto"  # TODO: test


class TopKInputs(BaseConceptInterpretationMethod):
    def __init__(
        self,
        *,
        granularity: Granularities,
        source: InterpretationSources,
        k: int = 5,
        model_with_split_points: ModelWithSplitPoints | None = None,  # TODO:
        split_point: str | None = None,
        concept_model: ConceptModel | None = None,
    ):
        super().__init__(model_with_split_points, split_point, concept_model)

        if source is InterpretationSources.VOCABULARY and granularity not in [
            # Granularities.ALL_TOKENS,  # TODO: add this granularity when implemented
            Granularities.TOKENS,
        ]:
            raise ValueError(f"The vocabulary granularity suppose a token granularity. Got {granularity}.")

        if granularity is not Granularities.TOKENS:
            raise NotImplementedError("Only token granularity is currently supported for interpretation.")

        self.granularity = granularity
        self.source = source
        self.k = k

    def verify_provided_sources(
        self,
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ):
        if inputs is None and self.source in [
            InterpretationSources.CONCEPTS_ACTIVATIONS,
            InterpretationSources.LATENT_ACTIVATIONS,
            InterpretationSources.INPUTS,
        ]:
            raise ValueError(f"The source {self.source} requires inputs to be provided. Please provide inputs.")

        if latent_activations is None and self.source is InterpretationSources.LATENT_ACTIVATIONS:
            raise ValueError(
                f"The source {self.source} requires latent activations to be provided. Please provide latent activations."
            )

        if concepts_activations is None and self.source is InterpretationSources.CONCEPTS_ACTIVATIONS:
            raise ValueError(
                f"The source {self.source} requires concepts activations to be provided. Please provide concepts activations."
            )

    def concepts_activations_from_source(
        self,
        inputs: list[str] | None = None,
        latent_activations: Float[torch.Tensor, "nl d"] | None = None,
        concepts_activations: Float[torch.Tensor, "nl cpt"] | None = None,
    ) -> tuple(list[str], Float[torch.Tensor, "nl cpt"]):
        # check that the provided sources are consistent with the arguments
        self.verify_provided_sources(inputs, latent_activations, concepts_activations)

        source = self.source
        if source is InterpretationSources.AUTO:
            if concepts_activations is not None:
                source = InterpretationSources.CONCEPTS_ACTIVATIONS
            elif latent_activations is not None:
                source = InterpretationSources.LATENT_ACTIVATIONS
            elif inputs is not None:
                source = InterpretationSources.INPUTS
            else:
                raise ValueError(
                    "The source `auto` requires either `concepts_activations`, `latent_activations` or `inputs` to be provided."
                )

        if source is InterpretationSources.VOCABULARY:
            inputs: list[str] = list(self.model_with_split_points.tokenizer.get_vocab().keys())

        if source in [
            InterpretationSources.VOCABULARY,
            InterpretationSources.INPUTS,
        ]:
            latent_activations: Float[torch.Tensor, "nl d"] = self.model_with_split_points.get_activations(
                inputs, select_strategy=ActivationSelectionStrategy.FLATTEN
            )[self.split_point]

        if source in [
            InterpretationSources.VOCABULARY,
            InterpretationSources.INPUTS,
            InterpretationSources.LATENT_ACTIVATIONS,
        ]:
            concepts_activations: Float[torch.Tensor, "nl cpt"] = self.concept_model.encode(latent_activations)

        return inputs, concepts_activations

    def granulated_inputs(
        self,
        inputs: list[str],  # (n, l)
        concepts_activations: ConceptsActivations,  # (n*l, cpt)
    ):
        max_seq_len = concepts_activations.shape[0] / len(inputs)

        if max_seq_len != int(max_seq_len):
            raise ValueError(
                f"The number of inputs and activations should be the same. Got {len(inputs)} inputs and {concepts_activations.shape[0]} activations."
            )
        max_seq_len = int(max_seq_len)

        if self.granularity is Granularities.TOKENS:
            indices_mask = torch.zeros(size=(concepts_activations.shape[0],), dtype=bool)

            granulated_flattened_inputs = []
            for i, input_example in enumerate(inputs):
                # TODO: check this treatment is correct, for now it has not really been tested
                tokens = self.model_with_split_points.tokenizer.tokenize(input_example)
                indices_mask[i * max_seq_len : i * max_seq_len + len(tokens)] = True
                granulated_flattened_inputs += tokens
            studied_inputs_concept_activations = concepts_activations[indices_mask]
        else:
            raise NotImplementedError(
                f"Granularity {self.granularity} is not yet implemented, only `TOKEN` is supported for now."
            )

        assert len(granulated_flattened_inputs) == len(studied_inputs_concept_activations)
        return granulated_flattened_inputs, studied_inputs_concept_activations

    def verify_concepts_indices(
        self,
        concepts_activations: ConceptsActivations,
        concepts_indices: int | list[int] | Literal["all"] = "all",
    ) -> list[int]:
        if concepts_indices == "all":
            concepts_indices = list(range(self.concept_model.nb_concepts))

        # take subset of concepts as specified by the user
        if isinstance(concepts_indices, int):
            concepts_indices = [concepts_indices]

        if not isinstance(concepts_indices, list) or not all(isinstance(c, int) for c in concepts_indices):
            raise ValueError(
                f"`concepts_indices` should be 'all', an int, or a list of int. Received {concepts_indices}."
            )

        if max(concepts_indices) >= concepts_activations.shape[1] or min(concepts_indices) < 0:
            raise ValueError(
                f"At least one concept index out of bounds. `max(concepts_indices)`: {max(concepts_indices)} >= {concepts_activations.shape[1]}."
            )

        return concepts_indices

    def topk_inputs_from_concepts_activations(
        self,
        inputs: list[str],  # (nl,)
        concepts_activations: ConceptsActivations,  # (nl, cpt)
        concepts_indices: list[int],  # TODO: sanitize this previously
    ) -> Mapping[int, Any]:
        # increase the number k to ensure that the top-k inputs are unique
        k = self.k * max(Counter(inputs).values())
        k = min(k, concepts_activations.shape[0])

        # Shape: (n*l, cpt_of_interest)
        concepts_activations = concepts_activations.T[concepts_indices].T

        # extract indices of the top-k input tokens for each specified concept
        topk_output = torch.topk(concepts_activations, k=k, dim=0)
        all_topk_activations = topk_output[0].T  # Shape: (cpt_of_interest, k)
        all_topk_indices = topk_output[1].T  # Shape: (cpt_of_interest, k)

        # create a dictionary with the interpretation
        interpretation_dict = {}
        # iterate over required concepts
        for cpt_idx, topk_activations, topk_indices in zip(
            concepts_indices, all_topk_activations, all_topk_indices, strict=True
        ):
            interpretation_dict[cpt_idx] = {}
            # iterate over k
            for activation, input_index in zip(topk_activations, topk_indices, strict=True):
                # ensure that the input is not already in the interpretation
                if len(interpretation_dict[cpt_idx]) >= self.k:
                    break
                if inputs[input_index] in interpretation_dict[cpt_idx]:
                    continue
                # set the kth input for the concept
                interpretation_dict[cpt_idx][inputs[input_index]] = activation.item()
        return interpretation_dict

    def interpret(
        self,
        concepts_indices: int | list[int] | Literal["all"] | None = "all",
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        # compute the concepts activations from the provided source, can also create inputs from the vocabulary
        inputs, concepts_activations = self.concepts_activations_from_source(
            inputs, latent_activations, concepts_activations
        )
        inputs: list[str]  # Verified by concepts_activations_from_source
        concepts_activations: Float[torch.Tensor, "nl cpt"]  # Verified by concepts_activations_from_source

        # inputs becomes a list of elements extracted from the examples
        # concepts_activations becomes a subset of the concepts activations corresponding to the inputs elements
        inputs, concepts_activations = self.granulated_inputs(inputs=inputs, concepts_activations=concepts_activations)

        concepts_indices = self.verify_concepts_indices(
            concepts_activations=concepts_activations, concepts_indices=concepts_indices
        )

        return self.topk_inputs_from_concepts_activations(
            inputs=inputs, concepts_activations=concepts_activations, concepts_indices=concepts_indices
        )
