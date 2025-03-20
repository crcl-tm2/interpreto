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
Tests for `interpreto.concepts.interpretation` methods
for `ConceptEncoderExplainer` and `ConceptAutoEncoderExplainer`
using the `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

from itertools import product

import pytest
import torch
from pytest import fixture
from transformers import AutoModelForMaskedLM

from interpreto.commons import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.concepts import NeuronsAsConcepts
from interpreto.concepts.interpretations import Granularities, InterpretationSources, TopKInputs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@fixture
def encoder_lm_splitter() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=[],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


def test_topk_inputs_from_activations():
    """
    Test that the `top_k_tokens_for_concept_from_activations` method works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    nb_concepts = 10
    n_tokens = 6
    k = 3
    n_samples = k * nb_concepts
    assert k <= n_tokens  # otherwise the test will break

    # generating data and fake activations
    # ['word_s0_t0', 'word_s0_t1',... 'word_s0_tn_tokens', 'word_s1_t0', 'word_s1_t1', ... 'word_s1_tn_tokens', ...]
    # sentences_list = [" ".join([f"word_s{i}_t{j}" for j in range(n_tokens)]) for i in range(n_samples)]
    words_list_list = [f"word_s{i}_t{j}" for i, j in product(range(n_samples), range(n_tokens))]
    fake_activations = torch.zeros(n_samples, n_tokens, nb_concepts)
    for i in range(nb_concepts):
        for j in range(k):
            fake_activations[i, (i + j) % n_tokens, i] = k - j
    fake_activations = fake_activations.view(-1, nb_concepts)

    # initializing the interpreter
    interpretation_method = TopKInputs(
        granularity=Granularities.TOKENS,
        source=InterpretationSources.CONCEPTS_ACTIVATIONS,
        k=k,
    )

    # extracting concept interpretations
    all_top_k_tokens = interpretation_method._topk_inputs_from_concepts_activations(
        inputs=words_list_list,
        concepts_activations=fake_activations,
        concepts_indices=list(range(nb_concepts)),
    )

    assert isinstance(all_top_k_tokens, dict) and len(all_top_k_tokens) == nb_concepts

    for c in range(nb_concepts):
        # correct format dict[int, list[tuple[str, float]]]
        assert isinstance(all_top_k_tokens[c], dict) and len(all_top_k_tokens[c]) == k

        # check the exact words returned
        for i, (w, a) in enumerate(all_top_k_tokens[c].items()):
            assert w == f"word_s{c}_t{(c + i) % n_tokens}"
            assert a == k - i

    indices = [0, 2, 4]
    subset_top_k_tokens = interpretation_method._topk_inputs_from_concepts_activations(
        inputs=words_list_list,
        concepts_activations=fake_activations,
        concepts_indices=indices,
    )

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = interpretation_method._topk_inputs_from_concepts_activations(
        inputs=words_list_list,
        concepts_activations=fake_activations,
        concepts_indices=[index],
    )
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]


def test_interpret_via_topk_inputs(encoder_lm_splitter: ModelWithSplitPoints):
    """
    Test that the `top_k_tokens_for_concept` method works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    hidden_size = 312
    n_tokens = 6
    k = 3
    n_samples = k * 5
    assert k <= n_tokens  # otherwise the test will break

    # generating data (these have no importance)
    joined_tokens_list = [" ".join([f"{i}{j}" for j in range(n_tokens)]) for i in range(n_samples)]
    larger_input = " ".join(["test" for _ in range(2 * n_tokens)])
    joined_tokens_list.append(larger_input)

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    encoder_lm_splitter.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=encoder_lm_splitter, split_point=split)

    # getting the activations
    activations = encoder_lm_splitter.get_activations(
        joined_tokens_list, select_strategy=ActivationSelectionStrategy.FLATTEN
    )

    # extracting concept interpretations
    all_top_k_tokens = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.LATENT_ACTIVATIONS,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
        latent_activations=activations,
    )

    assert isinstance(all_top_k_tokens, dict) and len(all_top_k_tokens) == hidden_size

    for c in range(hidden_size):
        # correct format dict[int, list[tuple[str, float]]]
        assert isinstance(all_top_k_tokens[c], dict) and len(all_top_k_tokens[c]) == k

    indices = [0, 2, 4]
    subset_top_k_tokens = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.LATENT_ACTIVATIONS,
        k=k,
        concepts_indices=indices,
        inputs=joined_tokens_list,
        latent_activations=activations,
    )

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.LATENT_ACTIVATIONS,
        k=k,
        concepts_indices=index,
        inputs=joined_tokens_list,
        latent_activations=activations,
    )
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]


def test_topk_inputs_sources(encoder_lm_splitter: ModelWithSplitPoints):
    """
    Test that different sources give the same results
    """
    hidden_size = 312
    n_tokens = 6
    k = 3
    n_samples = k * 5
    assert k <= n_tokens  # otherwise the test will break

    # generating data with duplicates and different lengths
    joined_tokens_list = [" ".join([f"{i + j}" for j in range(n_tokens)]) for i in range(n_samples)]
    larger_input = " ".join(["test" for _ in range(2 * n_tokens)])
    joined_tokens_list.append(larger_input)

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    encoder_lm_splitter.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=encoder_lm_splitter, split_point=split)

    # getting the activations
    activations = encoder_lm_splitter.get_activations(
        joined_tokens_list, select_strategy=ActivationSelectionStrategy.FLATTEN
    )

    # getting the top k tokens
    top_k_inputs = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.INPUTS,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
    )
    top_k_latent = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.LATENT_ACTIVATIONS,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
        latent_activations=activations,
    )
    top_k_concept = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.CONCEPTS_ACTIVATIONS,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
        concepts_activations=activations[split],
    )

    assert list(range(hidden_size)) == list(top_k_latent.keys())
    assert list(range(hidden_size)) == list(top_k_concept.keys())
    assert list(range(hidden_size)) == list(top_k_inputs.keys())

    for top_latent, top_concept, top_input in zip(
        top_k_latent.values(), top_k_concept.values(), top_k_inputs.values(), strict=True
    ):
        assert len(top_latent) == k
        assert top_latent == top_concept == top_input

    hidden_size = 312
    nb_concepts = 10
    top_k_vocabulary = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        granularity=Granularities.TOKENS,
        source=InterpretationSources.VOCABULARY,
        k=k,
        concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
    )

    assert len(top_k_vocabulary) == nb_concepts

    vocabulary = list(encoder_lm_splitter.tokenizer.get_vocab().keys())
    for topk in top_k_vocabulary.values():
        assert len(topk) == k
        for token in topk:
            assert token in vocabulary


def test_topk_inputs_error_raising(encoder_lm_splitter: ModelWithSplitPoints):
    """
    Test that the `TopKInputs` class raises an error when needed
    """
    # generating data (these have no importance)
    testing_examples = ["a sentence", "another sentence", "yet another sentence"]

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    encoder_lm_splitter.split_points = split

    # getting the activations
    activations = encoder_lm_splitter.get_activations(testing_examples)[split]

    # incompatible granularity and source  # TODO: add this test when Granularities.WORDS is implemented
    # with pytest.raises(ValueError):
    #     TopKInputs(
    #         InterpretationSources.VOCABULARY,
    #         granularity=Granularities.WORDS,
    #     )

    # source inputs but inputs is not provided
    with pytest.raises(ValueError):
        method = TopKInputs(
            source=InterpretationSources.INPUTS,
            granularity=Granularities.TOKENS,
        )
        method.interpret(
            concepts_indices="all",
            latent_activations=activations,
            concepts_activations=activations,
        )

    # source latent activations but latent activations is not provided
    with pytest.raises(ValueError):
        method = TopKInputs(
            source=InterpretationSources.LATENT_ACTIVATIONS,
            granularity=Granularities.TOKENS,
        )
        method.interpret(
            concepts_indices="all",
            inputs=testing_examples,
            concepts_activations=activations,
        )

    # source concepts activations but concepts activations is not provided
    with pytest.raises(ValueError):
        method = TopKInputs(
            source=InterpretationSources.CONCEPTS_ACTIVATIONS,
            granularity=Granularities.TOKENS,
        )
        method.interpret(
            concepts_indices="all",
            inputs=testing_examples,
            latent_activations=activations,
        )

    # wrong indices
    for wrong_indices in [-1, activations.shape[1], [-1, 0, 1], ["?"], (0, 1, 2), "str other than 'all'"]:
        with pytest.raises(ValueError):
            method = TopKInputs(
                source=InterpretationSources.CONCEPTS_ACTIVATIONS,
                granularity=Granularities.TOKENS,
                model_with_split_points=encoder_lm_splitter,
            )
            method.interpret(
                concepts_indices=wrong_indices,
                inputs=testing_examples,
                concepts_activations=activations,
            )

    # incompatible inputs and concepts activations $10 / 3$ is not an integer
    with pytest.raises(ValueError):
        method = TopKInputs(
            source=InterpretationSources.CONCEPTS_ACTIVATIONS,
            granularity=Granularities.TOKENS,
        )
        method._granulated_inputs(
            inputs=["one", "two", "three"],
            concepts_activations=torch.rand(10, 10),
        )
