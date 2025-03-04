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
Tests for interpreto.concepts.methods.concept_bottleneck methods
"""

from __future__ import annotations

import pytest
import torch
from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae
from torch import nn
from torch.nn import functional as F

from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.methods.overcomplete_cbe import (
    OvercompleteDictionaryLearning,
    OvercompleteMethods,
    OvercompleteSAE,
)

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


def test_verify_activations():
    """
    Test verify_activations
    """
    n_samples = 20
    input_size = 5
    hidden_size = 10
    n_concepts = 7

    model = EasilySplittableModel(input_size=input_size, hidden_size=hidden_size, output_size=2)
    split = "input_to_latent"
    splitted_model = ModelSplitterPlaceholder(model, split)

    inputs = torch.randn(n_samples, input_size)

    # different splits between activations and model
    activations = {
        "input_to_latent": splitted_model.get_activations(inputs),
        "end_model": inputs,
    }
    with pytest.raises(ValueError):
        cbe = OvercompleteDictionaryLearning(splitted_model, oc_opt.NMF, n_concepts=n_concepts)
        cbe.fit(activations, split="input_to_latent")

    # several splits in activations and model but none specified
    splitted_model.splits = ["input_to_latent", "end_model"]
    # conflict with activations
    with pytest.raises(ValueError):
        cbe = OvercompleteDictionaryLearning(splitted_model, oc_opt.NMF, n_concepts=n_concepts)
        cbe.fit(activations)
    # conflict with activations
    with pytest.raises(ValueError):
        cbe = OvercompleteDictionaryLearning(splitted_model, oc_opt.NMF, n_concepts=n_concepts)
        cbe.fit(activations[split])


@pytest.mark.slow
def test_overcomplete_cbe():
    """
    Test OvercompleteSAE and OvercompleteDictionaryLearning
    """
    n_samples = 20
    input_size = 5
    hidden_size = 10
    n_concepts = 7

    model = EasilySplittableModel(input_size=input_size, hidden_size=hidden_size, output_size=2)
    split = "input_to_latent"
    splitted_model = ModelSplitterPlaceholder(model, split)

    inputs = torch.randn(n_samples, input_size)
    activations = splitted_model.get_activations(inputs)
    assert activations[split].shape == (n_samples, hidden_size)

    # iterate over all methods from the namedtuple listing them
    for method in OvercompleteMethods:
        if issubclass(method.value, oc_sae.SAE):
            cbe = OvercompleteSAE(splitted_model, method.value, n_concepts=n_concepts, device=DEVICE)
            cbe.fit(activations, nb_epochs=1, batch_size=n_samples // 2, device=DEVICE)
        else:
            cbe = OvercompleteDictionaryLearning(
                splitted_model,
                method.value,
                n_concepts=n_concepts,
                device=DEVICE,
            )
            cbe.fit(activations)

        assert hasattr(cbe, "concept_encoder_decoder")
        assert hasattr(cbe, "splitted_model")
        assert cbe.fitted
        assert cbe.split == split
        assert hasattr(cbe, "_differentiable_concept_encoder")
        assert hasattr(cbe, "_differentiable_concept_decoder")

        concepts = cbe.encode_activations(activations)
        assert concepts.shape == (n_samples, n_concepts)
        reconstructed_activations = cbe.decode_concepts(concepts)
        assert reconstructed_activations.shape == (n_samples, hidden_size)


test_overcomplete_cbe()  # TODO: remove
