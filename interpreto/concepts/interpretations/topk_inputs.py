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
from typing import Any

import torch
from jaxtyping import Float
from nnsight.intervention.graph import InterventionProxy

from interpreto.commons import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.concepts.interpretations.base import BaseConceptInterpretationMethod
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations


class Granularities(Enum):  # TODO: harmonize with attribution granularities
    """Code [:octicons-mark-github-24: `concepts/interpretations/topk_inputs.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/topk_inputs.py)

    Possible granularities of inputs returned by the Top-K Inputs concept interpretation method.

    Valid granularities are:

    - `TOKENS`: the granularity is at the token level.
    """

    # ALL_TOKENS = "all_tokens"
    TOKENS = "tokens"
    # WORDS = "words"
    # CLAUSES = "clauses"
    # SENTENCES = "sentences"


class InterpretationSources(Enum):
    """Code [:octicons-mark-github-24: `concepts/interpretations/topk_inputs.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/topk_inputs.py)

    Possible sources of inputs to use for the Top-K Inputs concept interpretation method.
    The activations do not need to take into account the granularity of the inputs. It is managed internally.

    Valid sources are:

    - `CONCEPTS_ACTIVATIONS`: also require `inputs` to return strings but assume that the `concepts_activations` are provided and correspond to the inputs. Hence it is the fastest source.

    - `LATENT_ACTIVATIONS`: also require `inputs` to return strings but assume that the `latent_activations` are provided and correspond to the inputs.
        The latent activations can be the one used to fit the `concepts_model`. Hence the easiest source to use.

    - `INPUTS`: requires `inputs` and compute activations on them to extract the most activating inputs. It is the slowest source.

    - `VOCABULARY`: each token of the tokenizer vocabulary is considered as an `inputs`, then activations are computed. This source has the least requirements.

    - `AUTO`: depending on the provided arguments, it will select the most appropriate source. Order of preference is:
        1. `CONCEPTS_ACTIVATIONS`
        2. `LATENT_ACTIVATIONS`
        3. `INPUTS`
        4. `VOCABULARY`
    """

    CONCEPTS_ACTIVATIONS = "concepts_activations"
    LATENT_ACTIVATIONS = "latent_activations"
    INPUTS = "inputs"
    VOCABULARY = "vocabulary"
    AUTO = "auto"  # TODO: test


class TopKInputs(BaseConceptInterpretationMethod):
    """Code [:octicons-mark-github-24: `concepts/interpretations/topk_inputs.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/topk_inputs.py)

    Implementation of the Top-K Inputs concept interpretation method also called MaxAct.
    It is the most natural way to interpret a concept, as it is the most natural way to explain a concept.
    Hence several papers used it without describing it.
    Nonetheless, we can reference Bricken et al. (2023) [^1] from Anthropic for their post on transformer-circuits.

    [^1]:
        Trenton Bricken*, Adly Templeton*, Joshua Batson*, Brian Chen*, Adam Jermyn*, Tom Conerly, Nicholas L Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, Chris Olah
        [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
        Transformer Circuits, 2023.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model with split points to use for the interpretation.
        split_point (str): The split point to use for the interpretation.
        concept_model (ConceptModelProtocol): The concept model to use for the interpretation.
        granularity (Granularities): The granularity at which the interpretation is computed.
            Ignored for source `VOCABULARY`.
        source (InterpretationSources): The source of the inputs to use for the interpretation.
        k (int): The number of inputs to use for the interpretation.
    """

    def __init__(
        self,
        *,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        granularity: Granularities,
        source: InterpretationSources,
        split_point: str | None = None,
        k: int = 5,
    ):
        super().__init__(
            model_with_split_points=model_with_split_points, concept_model=concept_model, split_point=split_point
        )

        if granularity is not Granularities.TOKENS:
            raise NotImplementedError("Only token granularity is currently supported for interpretation.")

        if source not in InterpretationSources:
            raise ValueError(f"The source {source} is not supported. Supported sources: {InterpretationSources}")

        self.granularity = granularity
        self.source = source
        self.k = k

    def _concepts_activations_from_source(
        self,
        inputs: list[str] | None = None,
        latent_activations: Float[torch.Tensor, "nl d"] | None = None,
        concepts_activations: Float[torch.Tensor, "nl cpt"] | None = None,
    ) -> tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
        # determine the automatic source
        source = self.source
        if source is InterpretationSources.AUTO:
            if concepts_activations is not None:
                source = InterpretationSources.CONCEPTS_ACTIVATIONS
            elif latent_activations is not None:
                source = InterpretationSources.LATENT_ACTIVATIONS
            elif inputs is not None:
                source = InterpretationSources.INPUTS
            else:
                source = InterpretationSources.VOCABULARY

        # vocabulary source: construct the inputs from the vocabulary and compute the latent activations
        if source is InterpretationSources.VOCABULARY:
            # extract and sort the vocabulary
            vocab_dict: dict[str, int] = self.model_with_split_points.tokenizer.get_vocab()
            input_ids: list[int]
            inputs, input_ids = zip(*vocab_dict.items(), strict=True)  # type: ignore

            # compute the vocabulary's latent activations
            input_tensor: Float[torch.Tensor, "v 1"] = torch.tensor(input_ids).unsqueeze(1)
            activations_dict: InterventionProxy = self.model_with_split_points.get_activations(
                input_tensor, select_strategy=ActivationSelectionStrategy.FLATTEN
            )  # TODO: verify `ModelWithSplitPoints.get_activations()` can take in ids
            latent_activations = self.model_with_split_points.get_split_activations(
                activations_dict, split_point=self.split_point
            )

        # not vocabulary source: ensure that the inputs are provided
        if inputs is None:
            raise ValueError(f"The source {self.source} requires inputs to be provided. Please provide inputs.")

        # inputs source: compute the latent activations from the inputs
        if source is InterpretationSources.INPUTS:
            activations_dict: InterventionProxy = self.model_with_split_points.get_activations(
                inputs, select_strategy=ActivationSelectionStrategy.FLATTEN
            )
            latent_activations = self.model_with_split_points.get_split_activations(
                activations_dict, split_point=self.split_point
            )

        # latent activation source: ensure that the latent activations are provided
        if source is InterpretationSources.LATENT_ACTIVATIONS:
            if latent_activations is None:
                raise ValueError(
                    f"The source {self.source} requires latent activations to be provided. Please provide latent activations."
                )

        # not concepts activation source: compute the concepts activations from the latent activations
        if source in [
            InterpretationSources.VOCABULARY,
            InterpretationSources.INPUTS,
            InterpretationSources.LATENT_ACTIVATIONS,
        ]:
            concepts_activations = self.concept_model.encode(latent_activations)  # type: ignore

        # concepts activation source: ensure that the concepts activations are provided
        if concepts_activations is None:
            raise ValueError(
                f"The source {self.source} requires concepts activations to be provided. Please provide concepts activations."
            )

        return inputs, concepts_activations

    def _get_granular_inputs(
        self,
        inputs: list[str],  # (n, l)
        concepts_activations: ConceptsActivations,  # (n*l, cpt)
    ):
        if self.source is InterpretationSources.VOCABULARY:
            # no granularity is needed
            return inputs, concepts_activations

        max_seq_len = concepts_activations.shape[0] / len(inputs)

        if max_seq_len != int(max_seq_len):
            raise ValueError(
                f"The number of inputs and activations should be the same. Got {len(inputs)} inputs and {concepts_activations.shape[0]} activations."
            )
        max_seq_len = int(max_seq_len)
        if self.granularity is Granularities.TOKENS:
            indices_mask = torch.zeros(size=(concepts_activations.shape[0],), dtype=torch.bool)

            granular_flattened_inputs = []
            for i, input_example in enumerate(inputs):
                # TODO: check this treatment is correct, for now it has not really been tested
                tokens = self.model_with_split_points.tokenizer.tokenize(input_example)
                indices_mask[i * max_seq_len : i * max_seq_len + len(tokens)] = True
                granular_flattened_inputs += tokens
            studied_inputs_concept_activations = concepts_activations[indices_mask]
        else:
            raise NotImplementedError(
                f"Granularity {self.granularity} is not yet implemented, only `TOKEN` is supported for now."
            )

        assert len(granular_flattened_inputs) == len(studied_inputs_concept_activations)
        return granular_flattened_inputs, studied_inputs_concept_activations

    def _verify_concepts_indices(
        self,
        concepts_activations: ConceptsActivations,
        concepts_indices: int | list[int],
    ) -> list[int]:
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

    def _topk_inputs_from_concepts_activations(
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
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a list of inputs allowing to interpret them.
        The granularity of input examples is determined by the `granularity` class attribute.

        The returned inputs are the most activating inputs for the concepts.

        The required arguments depend on the `source` class attribute.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (list[str] | None): The inputs to use for the interpretation.
                Necessary if the source is not `VOCABULARY`, as examples are extracted from the inputs.
            latent_activations (Float[torch.Tensor, "nl d"] | None): The latent activations to use for the interpretation.
                Necessary if the source is `LATENT_ACTIVATIONS`.
                Otherwise, it is computed from the inputs or ignored if the source is `CONCEPT_ACTIVATIONS`.
            concepts_activations (Float[torch.Tensor, "nl cpt"] | None): The concepts activations to use for the interpretation.
                Necessary if the source is not `CONCEPT_ACTIVATIONS`. Otherwise, it is computed from the latent activations.

        Returns:
            Mapping[int, Any]: The interpretation of the concepts indices.

        Raises:
            ValueError: If the arguments do not correspond to the specified source.
        """
        # compute the concepts activations from the provided source, can also create inputs from the vocabulary
        sure_inputs: list[str]  # Verified by concepts_activations_from_source
        sure_concepts_activations: Float[torch.Tensor, "nl cpt"]  # Verified by concepts_activations_from_source
        sure_inputs, sure_concepts_activations = self._concepts_activations_from_source(
            inputs, latent_activations, concepts_activations
        )

        granular_inputs: list[str]  # len: ng, inputs becomes a list of elements extracted from the examples
        granular_concepts_activations: Float[torch.Tensor, "ng cpt"]
        granular_inputs, granular_concepts_activations = self._get_granular_inputs(
            inputs=sure_inputs, concepts_activations=sure_concepts_activations
        )

        concepts_indices = self._verify_concepts_indices(
            concepts_activations=granular_concepts_activations, concepts_indices=concepts_indices
        )

        return self._topk_inputs_from_concepts_activations(
            inputs=granular_inputs,
            concepts_activations=granular_concepts_activations,
            concepts_indices=concepts_indices,
        )
