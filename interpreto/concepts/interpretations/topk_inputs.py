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

import torch
from jaxtyping import Float

from interpreto.commons import ModelWithSplitPoints
from interpreto.concepts.interpretations.base import BaseConceptInterpretationMethod, verify_concepts_indices
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
        k (int): The number of inputs to use for the interpretation.
    """

    def __init__(
        self,
        *,
        model_with_split_points: ModelWithSplitPoints,
        split_point: str,
        concept_model: ConceptModelProtocol,
        granularity: Granularities,
        k: int = 5,
    ):
        super().__init__(model_with_split_points, split_point, concept_model)

        if granularity is not Granularities.TOKENS:
            raise NotImplementedError("Only token granularity is currently supported for interpretation.")

        self.granularity = granularity
        self.k = k

    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
        use_vocab: bool = False,
    ) -> Mapping[int, Mapping[str, float]]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a list of inputs allowing to interpret them.
        The granularity of input examples is determined by the `granularity` class attribute.

        The returned inputs are the most activating inputs for the concepts.

        The required arguments depend on the `source` class attribute.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (list[str] | None): The inputs to use for the interpretation.
            latent_activations (Float[torch.Tensor, "nl d"] | None): The latent activations to use for the interpretation.
            concepts_activations (Float[torch.Tensor, "nl cpt"] | None): The concepts activations to use for the interpretation.
            use_vocab (bool): Whether to use the vocabulary for the interpretation.

        Returns:
            Mapping[int, Any]: The interpretation of the concepts indices.

        Raises:
            ValueError: If the arguments do not correspond to the specified source.
        """
        # compute the concepts activations from the provided source, can also create inputs from the vocabulary
        sure_inputs: list[str]  # Verified by concepts_activations_from_source
        sure_concepts_activations: Float[torch.Tensor, "nl cpt"]  # Verified by concepts_activations_from_source
        sure_inputs, sure_concepts_activations = self.concepts_activations_from_source(
            inputs=inputs,
            latent_activations=latent_activations,
            concepts_activations=concepts_activations,
            use_vocab=use_vocab,
        )

        granular_inputs: list[str]  # len: ng, inputs becomes a list of elements extracted from the examples
        granular_concepts_activations: Float[torch.Tensor, "ng cpt"]
        granular_inputs, granular_concepts_activations = self._get_granular_inputs(
            inputs=sure_inputs, concepts_activations=sure_concepts_activations, from_vocab=use_vocab
        )

        concepts_indices = verify_concepts_indices(
            concepts_activations=granular_concepts_activations, concepts_indices=concepts_indices
        )

        return self._topk_inputs_from_concepts_activations(
            inputs=granular_inputs,
            concepts_activations=granular_concepts_activations,
            concepts_indices=concepts_indices,
        )

    def _get_granular_inputs(
        self,
        inputs: list[str],  # (n, l)
        concepts_activations: ConceptsActivations,  # (n*l, cpt)
        from_vocab: bool,
    ) -> tuple[list[str], Float[torch.Tensor, "ng cpt"]]:
        if from_vocab:
            # no granularity is needed
            return inputs, concepts_activations

        if self.granularity is not Granularities.TOKENS:
            raise NotImplementedError(
                f"Granularity {self.granularity} is not yet implemented, only `TOKEN` is supported for now."
            )

        max_seq_len = concepts_activations.shape[0] / len(inputs)
        if max_seq_len != int(max_seq_len):
            raise ValueError(
                f"The number of inputs and activations should be the same. Got {len(inputs)} inputs and {concepts_activations.shape[0]} activations."
            )

        # Select tokens that correspond to text (= no padding) ?
        max_seq_len = int(max_seq_len)
        indices_mask = torch.zeros(size=(concepts_activations.shape[0],), dtype=torch.bool)
        granular_flattened_inputs: list[str] = []
        for i, input_example in enumerate(inputs):
            # TODO: check this treatment is correct, for now it has not really been tested
            tokens = self.model_with_split_points.tokenizer.tokenize(input_example)
            indices_mask[i * max_seq_len : i * max_seq_len + len(tokens)] = True
            granular_flattened_inputs += tokens
        studied_inputs_concept_activations = concepts_activations[indices_mask]

        assert len(granular_flattened_inputs) == len(studied_inputs_concept_activations)
        return granular_flattened_inputs, studied_inputs_concept_activations

    def _topk_inputs_from_concepts_activations(
        self,
        inputs: list[str],  # (nl,)
        concepts_activations: ConceptsActivations,  # (nl, cpt)
        concepts_indices: list[int],  # TODO: sanitize this previously
    ) -> Mapping[int, dict[str, float]]:
        # increase the number k to ensure that the top-k inputs are unique
        big_k = self.k * max(Counter(inputs).values())
        big_k = min(big_k, concepts_activations.shape[0])

        concepts_activations = concepts_activations.T[concepts_indices]  # Shape: (cpt_of_interest, n*l)
        all_topk_activations, all_topk_indices = torch.topk(
            concepts_activations, k=self.k, dim=1
        )  # Shape: (cpt_of_interest, k)

        interpretation_dict: Mapping[int, dict[str, float]] = {}
        for cpt_idx, topk_activations, topk_indices in zip(
            concepts_indices, all_topk_activations, all_topk_indices, strict=True
        ):
            interpretation_dict[cpt_idx] = {}
            for activation, input_index in zip(topk_activations, topk_indices, strict=True):
                if len(interpretation_dict[cpt_idx]) >= self.k:
                    break
                if inputs[input_index] in interpretation_dict[cpt_idx]:
                    continue
                interpretation_dict[cpt_idx][inputs[input_index]] = activation.item()
        return interpretation_dict
