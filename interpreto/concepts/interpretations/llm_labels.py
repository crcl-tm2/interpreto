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
    texts: list[str]
    activations: torch.Tensor


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
            examples = self._sample_examples(
                inputs=granular_inputs,
                concepts_activations=sure_concepts_activations,
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
        inputs: list[str],  # (ng)
        concepts_activations: ConceptsActivations,  # (ng, cpt)
        sample_ids: list[int],  # (ng)
        concept_idx: int,
    ) -> list[Example]:
        """
        Only token granularity suported for now
        Select self.k_examples from the tokens and get their context and text
        """

        if self.sampling_method == SAMPLING_METHOD.TOP:
            _, inputs_indices = torch.topk(concepts_activations[:, concept_idx], k=self.k_examples)
            # TODO : verify that there is no zero values
        elif self.sampling_method == SAMPLING_METHOD.QUANTILE:
            inputs_indices = _sample_quantile(concepts_activations[:, concept_idx], k_examples=self.k_examples)
        elif self.sampling_method == SAMPLING_METHOD.RANDOM:
            non_zero_samples = torch.argwhere(concepts_activations[:, concept_idx] != 0).squeeze(-1)
            inputs_indices = non_zero_samples[torch.randperm(len(non_zero_samples))][: self.k_examples]
        else:
            raise NotImplementedError(f"Sampling method {self.sampling_method} is not implemented.")

        max_act = concepts_activations[:, concept_idx].max().item()
        examples: list[Example] = []
        for input_idx in inputs_indices:
            left_idx = max(0, input_idx.item() - self.k_context)
            right_idx = input_idx + self.k_context + 1
            # Select context from the same sample
            sample_idx = sample_ids[input_idx]
            example = Example(
                texts=[
                    text
                    for text, id in zip(inputs[left_idx:right_idx], sample_ids[left_idx:right_idx], strict=False)
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
            examples.append(example)
        return examples


def _sample_quantile(
    concept_activations: Float[torch.Tensor, "ng"], k_examples: int, n_quantiles: int = 5
) -> torch.Tensor:
    # TODO : add some checks on number of samples vs number of quantiles ect
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )

    non_zero_samples = torch.argwhere(concept_activations != 0)
    quantile_size = non_zero_samples.size(0) // n_quantiles  # TODO : some samples might be left out
    samples_per_quantile = k_examples // n_quantiles

    sorted_indexes = torch.argsort(concept_activations, descending=True)[: non_zero_samples.size(0)]
    sample_indices = torch.zeros(samples_per_quantile * n_quantiles)
    for i in range(n_quantiles):
        quantile_samples = sorted_indexes[i * quantile_size : (i + 1) * quantile_size]
        selected_samples = quantile_samples[torch.randperm(len(quantile_samples))[:samples_per_quantile]]
        sample_indices[i * samples_per_quantile : (i + 1) * samples_per_quantile] = selected_samples
    return sample_indices.long()


def _build_prompt(examples: list[Example]) -> list[tuple[Role, str]]:
    """
    Examples are given like that (example with tokens, but works similarly with words or sentences):

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
