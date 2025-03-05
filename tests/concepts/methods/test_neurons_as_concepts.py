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
Tests for `NeuronsAsConcepts` concept explainer
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


def test_identity_concepts_encoding_decoding():
    """
    Test that the concept encoding and decoding of the `NeuronsAsConcepts` is the identity
    """
    n_samples = 20
    input_size = 5
    hidden_size = 10

    model = EasilySplittableModel(input_size=input_size, hidden_size=hidden_size, output_size=2)
    split = "input_to_latent"
    splitted_model = ModelSplitterPlaceholder(model, split)

    inputs = torch.randn(n_samples, input_size)
    activations = splitted_model.get_activations(inputs)[split]

    concept_explainer = NeuronsAsConcepts(splitted_model)

    assert concept_explainer.fitted is True  # splitted_model has a single split so it is fitted
    assert concept_explainer.split == split
    assert hasattr(concept_explainer, "_differentiable_concept_encoder")
    assert hasattr(concept_explainer, "_differentiable_concept_decoder")

    concepts = concept_explainer.encode_activations(activations)
    reconstructed_activations = concept_explainer.decode_concepts(concepts)

    assert torch.allclose(concepts, activations)
    assert torch.allclose(reconstructed_activations, activations)
