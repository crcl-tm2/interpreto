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

import logging
from collections.abc import Mapping
from enum import Enum
from typing import NamedTuple

import torch
from jaxtyping import Float

from interpreto import Granularity, ModelWithSplitPoints
from interpreto.concepts.interpretations.base import BaseConceptInterpretationMethod, verify_concepts_indices
from interpreto.model_wrapping.llm_interface import LLMInterface, Role
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations


class SAMPLING_METHOD(Enum):
    TOP = "top"
    QUANTILE = "quantile"
    RANDOM = "random"


class Example(NamedTuple):
    texts: list[str] | str
    activations: list[int] | int


class LLMLabels(BaseConceptInterpretationMethod):
    """Code [:octicons-mark-github-24: `concepts/interpretations/llm_labels.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/llm_labels.py)

    Implement the automatic labeling method using a language model (LLM) to provide a short textual description given some examples of what activate the concept.
    This method was first introduced in [^1], we implement here the step 1 of the method.

    [^1]:
        Steven Bills*, Nick Cammarata*, Dan Mossing*, Henk Tillman*, Leo Gao*, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu*, William Saunders*
        [Language models can explain neurons in language models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
        2023.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model with split points to use for the interpretation.
        split_point (str): The split point to use for the interpretation.
        concept_model (ConceptModelProtocol): The concept model to use for the interpretation.
        activation_granularity (ActivationGranularity): The granularity at which the interpretation is computed.
            Allowed values are `TOKEN`, `WORD`, `SENTENCE`, and `SAMPLE`.
            Ignored when use_vocab=True.
        llm_interface (LLMInterface): The LLM interface to use for the interpretation.
        sampling_method (SAMPLING_METHOD): The method to use for sampling the inputs provided to the LLM.
        k_examples (int): The number of inputs to use for the interpretation.
        k_context (int): The number of context tokens to use around the concept tokens.
        use_vocab (bool): If True, the interpretation will be computed from the vocabulary of the model.
    """

    def __init__(
        self,
        *,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        split_point: str | None = None,
        activation_granularity: ActivationGranularity = ActivationGranularity.TOKEN,
        llm_interface: LLMInterface,
        sampling_method: SAMPLING_METHOD,
        k_examples: int = 30,
        k_context: int = 10,
        use_vocab: bool = False,
        k_quantile: int = 5,
    ):
        super().__init__(
            model_with_split_points=model_with_split_points, split_point=split_point, concept_model=concept_model
        )

        if activation_granularity not in (
            ActivationGranularity.TOKEN,
            ActivationGranularity.WORD,
            ActivationGranularity.SENTENCE,
            ActivationGranularity.SAMPLE,
        ):
            raise ValueError(
                f"The granularity {activation_granularity} is not supported. Supported `activation_granularities`: TOKEN, WORD, SENTENCE, and SAMPLE"
            )
        self.activation_granularity = activation_granularity

        self.llm_interface = llm_interface
        self.sampling_method = sampling_method
        self.k_examples = k_examples
        self.k_context = k_context
        self.use_vocab = use_vocab
        self.k_quantile = k_quantile

    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, str | None]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a short textual description.
        The granularity of input examples is determined by the `activation_granularity` class attribute.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (list[str] | None): The inputs to use for the interpretation.
                Necessary if not `use_vocab`, as examples are extracted from the inputs.
            latent_activations (Float[torch.Tensor, "nl d"] | None): The latent activations matching the inputs. If not provided, it is computed from the inputs.
            concepts_activations (Float[torch.Tensor, "nl cpt"] | None): The concepts activations matching the inputs. If not provided, it is computed from the inputs or latent activations.
        Returns:
            Mapping[int, str | None]: The textual labels of the concepts indices.
        """

        if self.use_vocab:
            sure_inputs, sure_concepts_activations = self.concepts_activations_from_vocab()
        else:
            if inputs is None:
                raise ValueError("Inputs must be provided when use_vocab is False.")
            sure_inputs = inputs
            sure_concepts_activations = self.concepts_activations_from_source(
                inputs=inputs,
                latent_activations=latent_activations,
                concepts_activations=concepts_activations,
            )

        concepts_indices = verify_concepts_indices(
            concepts_activations=sure_concepts_activations,
            concepts_indices=concepts_indices,
        )

        granular_inputs, granular_sample_ids = self._get_granular_inputs(
            inputs=sure_inputs,
            concepts_activations=sure_concepts_activations,
        )

        labels: Mapping[int, str | None] = {}
        for concept_idx in concepts_indices:
            example_idx = self._sample_examples(
                concepts_activations=sure_concepts_activations,
                concept_idx=concept_idx,
            )
            examples = _format_examples(
                example_ids=example_idx,
                inputs=granular_inputs,
                concept_activations=sure_concepts_activations[:, concept_idx],
                sample_ids=granular_sample_ids,
                k_context=self.k_context,
            )
            example_prompt = _build_example_prompt(examples)
            prompt = [(Role.SYSTEM, SYSTEM_PROMPT), (Role.USER, example_prompt), (Role.ASSISTANT, "")]
            labels[concept_idx] = self.llm_interface.generate(prompt)
        return labels

    def _get_granular_inputs(
        self,
        inputs: list[str],  # (n)
        concepts_activations: ConceptsActivations,  # (n*l, cpt)
    ) -> tuple[list[str], list[int]]:
        max_seq_len = concepts_activations.shape[0] / len(inputs)
        if max_seq_len != int(max_seq_len):
            raise ValueError(
                f"The number of inputs and activations should be the same. Got {len(inputs)} inputs and {concepts_activations.shape[0]} activations."
            )

        if self.use_vocab or self.activation_granularity is ActivationGranularity.SAMPLE:
            # no activation_granularity is needed
            return inputs, list(range(len(inputs)))

        # Get granular texts from the inputs
        tokens = self.model_with_split_points.tokenizer(
            inputs, return_tensors="pt", padding=True, return_offsets_mapping=True
        )
        granular_texts: list[list[str]] = Granularity.get_decomposition(
            tokens,
            granularity=self.activation_granularity.value,  # type: ignore
            tokenizer=self.model_with_split_points.tokenizer,
            return_text=True,
        )  # type: ignore

        granular_flattened_texts = [text for sample_texts in granular_texts for text in sample_texts]
        granular_flattened_sample_id = [i for i, sample_texts in enumerate(granular_texts) for _ in sample_texts]
        return granular_flattened_texts, granular_flattened_sample_id

    def _sample_examples(
        self,
        concepts_activations: ConceptsActivations,  # (ng, cpt)
        concept_idx: int,
    ) -> list[int]:
        if self.sampling_method == SAMPLING_METHOD.TOP:
            inputs_idx = _sample_top(
                concept_activations=concepts_activations[:, concept_idx],
                k_examples=self.k_examples,
            )
        elif self.sampling_method == SAMPLING_METHOD.QUANTILE:
            inputs_idx = _sample_quantile(
                concept_activations=concepts_activations[:, concept_idx],
                k_examples=self.k_examples,
                k_quantile=self.k_quantile,
            )
        elif self.sampling_method == SAMPLING_METHOD.RANDOM:
            inputs_idx = _sample_random(
                concept_activations=concepts_activations[:, concept_idx],
                k_examples=self.k_examples,
            )
        else:
            raise NotImplementedError(f"Sampling method {self.sampling_method} is not implemented.")
        return inputs_idx


def _sample_top(
    concept_activations: Float[torch.Tensor, "ng"],
    k_examples: int,
) -> list[int]:
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )
    non_zero_samples = torch.argwhere(concept_activations != 0).squeeze(-1)
    k_examples = min(k_examples, non_zero_samples.size(0))
    inputs_indices = non_zero_samples[torch.topk(concept_activations[non_zero_samples], k=k_examples).indices]
    return inputs_indices.tolist()


def _sample_random(
    concept_activations: Float[torch.Tensor, "ng"],
    k_examples: int,
) -> list[int]:
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )
    non_zero_samples = torch.argwhere(concept_activations != 0).squeeze(-1)
    inputs_indices = non_zero_samples[torch.randperm(len(non_zero_samples))][:k_examples]
    return inputs_indices.tolist()


def _sample_quantile(
    concept_activations: Float[torch.Tensor, "ng"], k_examples: int, k_quantile: int = 5
) -> list[int]:
    if k_examples < k_quantile:
        raise ValueError(f"k_examples ({k_examples}) should be greater than k_quantile ({k_quantile}).")
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )

    non_zero_samples = torch.argwhere(concept_activations != 0).squeeze(-1)
    if non_zero_samples.size(0) < k_quantile:
        logging.warning("Not enough non-zero samples to compute quantiles. Using all non-zero samples.")
        return non_zero_samples.tolist()

    quantile_size = non_zero_samples.size(0) // k_quantile
    samples_per_quantile = k_examples // k_quantile

    sorted_indexes = torch.argsort(concept_activations, descending=True)[: non_zero_samples.size(0)]
    sample_indices = []
    for i in range(k_quantile):
        if i == k_quantile - 1:
            # Last quantile (minimally activating samples) may have more samples
            quantile_samples = sorted_indexes[i * quantile_size :]
        else:
            quantile_samples = sorted_indexes[i * quantile_size : (i + 1) * quantile_size]
        selected_samples = quantile_samples[torch.randperm(len(quantile_samples))[:samples_per_quantile]]
        sample_indices.extend(selected_samples.tolist())
    return sample_indices


def _format_examples(
    example_ids: list[int],
    inputs: list[str],
    concept_activations: Float[torch.Tensor, "ng"],
    sample_ids: list[int],  # (ng)
    k_context: int,
) -> list[Example]:
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )
    if len(inputs) != len(sample_ids) or len(inputs) != concept_activations.size(0):
        raise ValueError(
            f"The number of inputs ({len(inputs)}), sample_ids ({len(sample_ids)}), and concept_activations ({concept_activations.size(0)}) should be the same."
        )

    max_act = concept_activations.max().item()
    examples: list[Example] = []
    for example_id in example_ids:
        if k_context > 0:
            left_idx = max(0, example_id - k_context)
            right_idx = example_id + k_context + 1
            # Select context from the same sample, it won't select tokens/sentences/texts from other samples
            sample_idx = sample_ids[example_id]
            example = Example(
                texts=[
                    text
                    for text, id in zip(inputs[left_idx:right_idx], sample_ids[left_idx:right_idx], strict=False)
                    if id == sample_idx
                ],
                activations=[
                    round(act.item() / max_act * 10)
                    for act, id in zip(
                        concept_activations[left_idx:right_idx],
                        sample_ids[left_idx:right_idx],
                        strict=False,
                    )
                    if id == sample_idx
                ],
            )
        else:
            example = Example(
                texts=inputs[example_id],
                activations=round(concept_activations[example_id].item() / max_act * 10),
            )
        examples.append(example)
    return examples


def _build_example_prompt(examples: list[Example]) -> str:
    """
    For examples provided with a context, the format is:

    Example 1:  the dog <<eats>> the cat
    Activations: ("the", 0), (" dog", 2), (" eats", 10), (" the", 2), (" cat", 0)
    Example 2:  it was a <<delicious>> meal, but
    Activations: ("it", 0), (" was", 0), (" a", 1), (" delicious", 9), (" meal", 8), (",", 0), (" but", 0)

    For examples without context, the format is:

    Example 1:  The dog eats the cat (activation : 6)
    Example 2:  it was a delicious meal (activation : 4)

    """
    example_prompts: list[str] = []
    for i, example in enumerate(examples):
        if isinstance(example.texts, str) and isinstance(example.activations, int):
            # Text without context
            example_prompts.append(f"Example {i + 1}: {example.texts} (activation: {example.activations})")
        elif isinstance(example.texts, list) and isinstance(example.activations, list):
            # Text with context
            max_text_pos = example.activations.index(max(example.activations))
            example_prompts.append(
                f"Example {i + 1}: "
                + "".join(example.texts[:max_text_pos])
                + f" <<{example.texts[max_text_pos]}>> "
                + "".join(example.texts[max_text_pos + 1 :])
            )
            example_prompts.append(
                "Activations: "
                + ", ".join(
                    [
                        f'("{text}", {activation})'
                        for text, activation in zip(example.texts, example.activations, strict=False)
                    ]
                )
            )
        else:
            raise ValueError(
                f"example.text is {type(example)} and example.activations is {example.texts}, expected str with int or list[str] with list[int]."
            )
    return "\n".join(example_prompts)


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
