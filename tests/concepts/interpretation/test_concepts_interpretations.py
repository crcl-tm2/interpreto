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
for `AbstractConceptExplainer` and `ConceptBottleneckExplainer`
using the `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

import torch
from pytest import fixture
from transformers import AutoModelForMaskedLM

from interpreto.commons.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.concepts import NeuronsAsConcepts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@fixture
def encoder_lm_splitter() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=[],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


def test_topk_inputs_from_activations(encoder_lm_splitter: ModelWithSplitPoints):
    """
    Test that the `top_k_tokens_for_concept_from_activations` method works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    input_size = 5
    hidden_size = 10
    n_tokens = 6
    k = 3
    n_samples = k * hidden_size
    assert k <= n_tokens  # otherwise the test will break

    # generating data and fake activations
    # ['word_s0_t0', 'word_s0_t1',... 'word_s0_tn_tokens', 'word_s1_t0', 'word_s1_t1', ... 'word_s1_tn_tokens', ...]
    # sentences_list = [" ".join([f"word_s{i}_t{j}" for j in range(n_tokens)]) for i in range(n_samples)]
    words_list_list = [[f"word_s{i}_t{j}" for j in range(n_tokens)] for i in range(n_samples)]
    fake_activations = torch.zeros(n_samples, n_tokens, hidden_size)
    for i in range(hidden_size):
        for j in range(k):
            fake_activations[i, (i + j) % n_tokens, i] = k - j
    fake_activations = fake_activations.view(-1, hidden_size)

    # initializing the explainer
    split = "bert.encoder.layer.1"
    encoder_lm_splitter.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=encoder_lm_splitter, split_point=split)

    # forcing the input_size of the concept model
    concept_explainer.concept_model.input_size = input_size

    # extracting concept interpretations
    all_top_k_tokens = concept_explainer.top_k_inputs_for_concept_from_activations(
        splitted_inputs=words_list_list, activations=fake_activations, k=k, concepts_indices="all"
    )

    assert isinstance(all_top_k_tokens, dict) and len(all_top_k_tokens) == hidden_size

    for c in range(hidden_size):
        # correct format dict[int, list[tuple[str, float]]]
        assert isinstance(all_top_k_tokens[c], list) and len(all_top_k_tokens[c]) == k

        # no duplicates
        assert len(all_top_k_tokens[c]) == len(set(all_top_k_tokens[c]))

        # check the exact words returned
        for i, w in enumerate(all_top_k_tokens[c]):
            assert isinstance(w, tuple) and len(w) == 2
            assert w[0] == f"word_s{c}_t{(c + i) % n_tokens}"
            assert w[1] == k - i

    indices = [0, 2, 4]
    subset_top_k_tokens = concept_explainer.top_k_inputs_for_concept_from_activations(
        splitted_inputs=words_list_list, activations=fake_activations, k=k, concepts_indices=indices
    )

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = concept_explainer.top_k_inputs_for_concept_from_activations(
        splitted_inputs=words_list_list, activations=fake_activations, k=k, concepts_indices=index
    )
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]


def test_topk_tokens(encoder_lm_splitter: ModelWithSplitPoints):
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
    split = "bert.encoder.layer.1"
    encoder_lm_splitter.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=encoder_lm_splitter, split_point=split)

    # extracting concept interpretations
    all_top_k_tokens = concept_explainer.top_k_inputs_for_concept(joined_tokens_list, k=k, concepts_indices="all")

    assert isinstance(all_top_k_tokens, dict) and len(all_top_k_tokens) == hidden_size

    for c in range(hidden_size):
        # correct format dict[int, list[tuple[str, float]]]
        assert isinstance(all_top_k_tokens[c], list) and len(all_top_k_tokens[c]) == k

        # no duplicates
        # assert len(all_top_k_tokens[c]) == len(set(all_top_k_tokens[c]))  # TODO: add this test back

    indices = [0, 2, 4]
    subset_top_k_tokens = concept_explainer.top_k_inputs_for_concept(joined_tokens_list, k=k, concepts_indices=indices)

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = concept_explainer.top_k_inputs_for_concept(joined_tokens_list, k=k, concepts_indices=index)
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]
