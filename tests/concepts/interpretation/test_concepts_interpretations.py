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
from torch import nn
from torch.nn import functional as F

from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts import NeuronsAsConcepts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EasilySplittableModel(nn.Module):
    """
    Dummy model with two parts for testing purposes
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def input_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def end_model(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc3(x))
        return F.relu(self.fc4(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_to_latent(x)
        return self.end_model(x)


def test_topk_tokens_from_activations():
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

    model = EasilySplittableModel(input_size=input_size, hidden_size=hidden_size, output_size=2)
    split = "input_to_latent"
    splitted_model = ModelSplitterPlaceholder(model, split)

    fake_activations = torch.zeros(n_samples, n_tokens, hidden_size)
    for i in range(hidden_size):
        for j in range(k):
            fake_activations[i, (i + j) % n_tokens, i] = k - j

    words_list_list = [[f"word_s{i}_t{j}" for j in range(n_tokens)] for i in range(n_samples)]

    concept_explainer = NeuronsAsConcepts(splitted_model)

    all_top_k_tokens = concept_explainer.top_k_tokens_for_concept_from_activations(
        inputs=words_list_list, activations=fake_activations, k=k, concepts_indices="all"
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
    subset_top_k_tokens = concept_explainer.top_k_tokens_for_concept_from_activations(
        inputs=words_list_list, activations=fake_activations, k=k, concepts_indices=indices
    )

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = concept_explainer.top_k_tokens_for_concept_from_activations(
        inputs=words_list_list, activations=fake_activations, k=k, concepts_indices=index
    )
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]


test_topk_tokens_from_activations()  # TODO: remove
