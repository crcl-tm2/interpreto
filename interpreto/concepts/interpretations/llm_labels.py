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


from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import NamedTuple

import torch
from jaxtyping import Float

from interpreto.commons import ModelWithSplitPoints
from interpreto.commons.model_wrapping.llm_interface import LLMInterface, Role
from interpreto.concepts.interpretations.base import BaseConceptInterpretationMethod, verify_concepts_indices
from interpreto.concepts.interpretations.topk_inputs import Granularities
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations


class SAMPLING_METHOD(Enum):
    TOP = "top"
    QUANTILE = "quantile"
    RANDOM = "random"


class Example(NamedTuple):
    tokens: list[str]
    activations: torch.Tensor


class LLMLabels(BaseConceptInterpretationMethod):
    """Code [:octicons-mark-github-24: `concepts/interpretations/llm_labels.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/llm_labels.py)

    TODO : describe the method and cite

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model with split points to use for the interpretation.
        split_point (str): The split point to use for the interpretation.
        concept_model (ConceptModelProtocol): The concept model to use for the interpretation.
        granularity (Granularities): The granularity at which the interpretation is computed.
        k (int): The number of inputs to use for the interpretation.
    """

    def __init__(
        self,
        *,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        split_point: str | None = None,
        granularity: Granularities,
        llm_interface: LLMInterface,
        sampling_method: SAMPLING_METHOD,
        k_examples: int = 30,
        k_context: int = 10,
    ):
        super().__init__(
            model_with_split_points=model_with_split_points, split_point=split_point, concept_model=concept_model
        )

        if granularity is not Granularities.TOKENS:
            raise NotImplementedError("Only token granularity is currently supported for interpretation.")
        self.granularity = granularity
        self.llm_interface = llm_interface
        self.sampling_method = sampling_method
        self.k_examples = k_examples
        self.k_context = k_context

    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
        use_vocab: bool = False,
    ) -> Mapping[int, str | None]:
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
        sure_inputs, sure_concepts_activations = self.concepts_activations_from_source(
            inputs=inputs,
            latent_activations=latent_activations,
            concepts_activations=concepts_activations,
            use_vocab=use_vocab,
        )

        concepts_indices = verify_concepts_indices(
            concepts_activations=sure_concepts_activations,
            concepts_indices=concepts_indices,
        )

        granular_inputs, granular_concepts_activations, granular_sample_ids = self._get_granular_inputs(
            inputs=sure_inputs,
            concepts_activations=sure_concepts_activations,
            from_vocab=use_vocab,
        )

        labels: Mapping[int, str | None] = {}
        for concept_idx in concepts_indices:
            examples = self._sample_examples(
                inputs=granular_inputs,
                concepts_activations=granular_concepts_activations,
                sample_ids=granular_sample_ids,
                concept_idx=concept_idx,
            )
            prompt = _build_prompt(examples)
            labels[concept_idx] = self.llm_interface.generate(prompt)
        return labels

    def _get_granular_inputs(
        self,
        inputs: list[str],  # (n)
        concepts_activations: ConceptsActivations,  # (n*l, cpt)
        from_vocab: bool,
    ) -> tuple[list[str], Float[torch.Tensor, "ng cpt"], list[int]]:
        """
        Only token granularity for now : Gets sequences of tokens and corresponding activations for non-padding tokens
        """

        if from_vocab:
            # no granularity is needed
            return inputs, concepts_activations, list(range(len(inputs)))

        if self.granularity is not Granularities.TOKENS:
            raise NotImplementedError(
                f"Granularity {self.granularity} is not yet implemented, only `TOKEN` is supported for now."
            )

        max_seq_len = concepts_activations.shape[0] / len(inputs)
        if max_seq_len != int(max_seq_len):
            raise ValueError(
                f"The number of inputs and activations should be the same. Got {len(inputs)} inputs and {concepts_activations.shape[0]} activations."
            )

        # Select tokens that correspond to text (= no padding) and keep track of their sample id
        max_seq_len = int(max_seq_len)
        indices_mask = torch.zeros(size=(concepts_activations.shape[0],), dtype=torch.bool)
        granular_flattened_inputs: list[str] = []
        granular_flattened_sample_id: list[int] = []
        for i, input_example in enumerate(inputs):
            # TODO: check this treatment is correct, for now it has not really been tested
            tokens = self.model_with_split_points.tokenizer.tokenize(input_example)
            indices_mask[i * max_seq_len : i * max_seq_len + len(tokens)] = True
            granular_flattened_inputs += tokens
            granular_flattened_sample_id += [i] * len(tokens)
        studied_inputs_concept_activations = concepts_activations[indices_mask]

        assert len(granular_flattened_inputs) == len(studied_inputs_concept_activations)
        return granular_flattened_inputs, studied_inputs_concept_activations, granular_flattened_sample_id

    def _sample_examples(
        self,
        inputs: list[str],  # (ng)
        concepts_activations: ConceptsActivations,  # (ng, cpt)
        sample_ids: list[int],  # (ng)
        concept_idx: int,
    ) -> list[Example]:
        """
        Select self.k_examples from the tokens and get their context and text
        """
        if self.granularity is not Granularities.TOKENS:
            raise NotImplementedError(
                f"Granularity {self.granularity} is not yet implemented, only `TOKEN` is supported for now."
            )

        if self.sampling_method == SAMPLING_METHOD.TOP:
            _, token_indices = torch.topk(concepts_activations[:, concept_idx], k=self.k_examples)
            # TODO : verify that there is no non-zero
        elif self.sampling_method == SAMPLING_METHOD.QUANTILE:
            token_indices = _sample_quantile(concepts_activations[:, concept_idx], k_examples=self.k_examples)
        elif self.sampling_method == SAMPLING_METHOD.RANDOM:
            non_zero_samples = torch.argwhere(concepts_activations[:, concept_idx] != 0).squeeze(-1)
            token_indices = non_zero_samples[torch.randperm(len(non_zero_samples))][: self.k_examples]
        else:
            raise NotImplementedError(f"Sampling method {self.sampling_method} is not implemented.")

        max_act = concepts_activations[:, concept_idx].max().item()
        tokens_examples: list[Example] = []
        for token_idx in token_indices:
            left_idx = max(0, token_idx.item() - self.k_context)
            right_idx = token_idx + self.k_context + 1
            # Select context tokens from the same sample
            sample_idx = sample_ids[token_idx]

            example = Example(
                tokens=[
                    tok
                    for tok, id in zip(inputs[left_idx:right_idx], sample_ids[left_idx:right_idx], strict=False)
                    if id == sample_idx
                ],
                activations=torch.tensor(
                    [
                        act / max_act * 10
                        for act, id in zip(
                            concepts_activations[left_idx:right_idx, concept_idx],
                            sample_ids[left_idx:right_idx],
                            strict=False,
                        )
                        if id == sample_idx
                    ]
                ).floor(),
            )
            tokens_examples.append(example)
        return tokens_examples


def _sample_quantile(max_per_input: Float[torch.Tensor, n], k_examples: int, n_quantiles: int = 5) -> torch.Tensor:
    # TODO : add some checks on number of samples vs number of quantiles ect
    if len(max_per_input.size()) > 1:
        raise ValueError(f"max_per_input should be a 1D tensor, got tensor of shape {max_per_input.size()}")

    non_zero_samples = torch.argwhere(max_per_input != 0)
    quantile_size = non_zero_samples.size(0) // n_quantiles  # TODO : some samples might be left out
    samples_per_quantile = k_examples // n_quantiles

    sorted_indexes = torch.argsort(max_per_input, descending=True)[: non_zero_samples.size(0)]
    sample_indices = torch.zeros(samples_per_quantile * n_quantiles)
    for i in range(n_quantiles):
        quantile_samples = sorted_indexes[i * quantile_size : (i + 1) * quantile_size]
        selected_samples = quantile_samples[torch.randperm(len(quantile_samples))[:samples_per_quantile]]
        sample_indices[i * samples_per_quantile : (i + 1) * samples_per_quantile] = selected_samples
    return sample_indices.long()


def _build_prompt(examples: list[Example]) -> list[tuple[Role, str]]:
    """
    Examples are given like that :

    Example 1:  the dog <<eats>> the cat
    Activations: ("the", 0), (" dog", 2), (" eats", 10), (" the", 2), (" cat", 0)
    Example 2:  it was a <<delicious>> meal, but
    Activations: ("it", 0), (" was", 0), (" a", 1), (" delicious", 9), (" meal", 8), (",", 0), (" but", 0)

    """
    example_prompts: list[str] = []
    for i, example in enumerate(examples):
        max_token = torch.argmax(example.activations)
        example_prompts.append(
            f"Example {i + 1}: "
            + "".join(example.tokens[:max_token])
            + f" <<{example.tokens[max_token]}>> "
            + "".join(example.tokens[max_token + 1 :])
        )
        example_prompts.append(
            "Activations: "
            + ", ".join(
                [
                    f'("{token}", {activation})'
                    for token, activation in zip(example.tokens, example.activations, strict=False)
                ]
            )
        )
    example_prompt = "\n".join(example_prompts)
    return [(Role.SYSTEM, SYSTEM_PROMPT), (Role.USER, example_prompt), (Role.ASSISTANT, "")]


# From https://github.com/EleutherAI/delphi/blob/article_version/sae_auto_interp/explainers/default/prompts.py
SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special tokens are selected and between delimiters like <<this>>.
How important each token is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations.
- Keep your explanations short and concise, with no more that 15 words, for example "reference to blue objects" or "word before a comma"
"""
