"""
Tests for interpreto.concepts.methods.concept_bottleneck methods
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.methods.overcomplete_cbe import OvercompleteCBE, OvercompleteMethods


class DummySplittedModel(nn.Module):
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


def test_overcomplete_cbe():
    """
    Test OvercompleteCBE
    """
    n_samples = 100
    input_size = 10
    hidden_size = 20
    n_concepts = hidden_size

    model = DummySplittedModel(input_size=input_size, hidden_size=hidden_size, output_size=2)
    splitted_model = ModelSplitterPlaceholder(model, "input_to_latent")

    inputs = torch.randn(n_samples, input_size)
    activations = splitted_model.get_activations(inputs)
    assert activations.shape == (n_samples, hidden_size)

    concept_extraction_methods_to_test = [OvercompleteMethods.NMF]  # TODO: add other methods

    for method in concept_extraction_methods_to_test:
        cbe = OvercompleteCBE(splitted_model, method, n_concepts=n_concepts)
        cbe.fit(activations)
        assert cbe.fitted
        assert hasattr(cbe, "_differentiable_concept_encoder")
        assert hasattr(cbe, "_differentiable_concept_decoder")

        concepts = cbe.encode_activations(activations)
        assert concepts.shape == (n_samples, n_concepts)
        reconstructed_activations = cbe.decode_concepts(concepts)
        assert reconstructed_activations.shape == (n_samples, hidden_size)
        assert torch.allclose(reconstructed_activations, activations, atol=1e-2)
