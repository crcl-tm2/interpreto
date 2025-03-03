import pytest
import torch
from transformers import BertModel, BertTokenizer

from interpreto.attributions.perturbations.base import GaussianNoisePerturbator
from interpreto.attributions.perturbations.linear_interpolation_perturbation import (
    LinearInterpolationPerturbation,
)
from interpreto.attributions.perturbations.occlusion import TokenOcclusionPerturbator, WordOcclusionPerturbator


def test_gaussian_noise_perturbator_perturb():
    perturbator = GaussianNoisePerturbator(n_perturbations=3, std=0.1)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perturbed_inputs, _ = perturbator.perturb(inputs)

    assert perturbed_inputs.shape == (3, 2, 2)
    assert perturbed_inputs.device == inputs.device
    assert torch.allclose(perturbed_inputs.mean(dim=0), inputs, atol=0.2)
    assert not torch.equal(perturbed_inputs, inputs)


def test_linear_interpolation_perturbation_perturb():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perturbator = LinearInterpolationPerturbation(n_perturbations=3, baseline=0.0)
    perturbed_inputs, _ = perturbator.perturb(inputs)

    assert perturbed_inputs.shape == (2, 3, 2)
    assert perturbed_inputs.device == inputs.device


def test_linear_interpolation_perturbation_perturb_with_tensor_baseline():
    baseline_tensor = torch.tensor([0.0, 0.0])
    perturbator = LinearInterpolationPerturbation(baseline=baseline_tensor, n_perturbations=3)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perturbed_inputs, _ = perturbator.perturb(inputs)

    assert perturbed_inputs.shape == (2, 3, 2)
    assert perturbed_inputs.device == inputs.device


def test_linear_interpolation_perturbation_adjust_baseline():
    inputs = torch.randn(4, 3, 10)

    # Test with None baseline
    baseline = LinearInterpolationPerturbation.adjust_baseline(None, inputs)
    assert torch.all(baseline == 0)
    assert baseline.shape == inputs.shape[1:]

    # Test with float baseline
    baseline = LinearInterpolationPerturbation.adjust_baseline(0.5, inputs)
    assert torch.all(baseline == 0.5)
    assert baseline.shape == inputs.shape[1:]

    # Test with tensor baseline
    baseline_tensor = torch.randn(3, 10)
    baseline = LinearInterpolationPerturbation.adjust_baseline(baseline_tensor, inputs)
    assert torch.all(baseline == baseline_tensor)
    assert baseline.shape == inputs.shape[1:]


def test_linear_interpolation_perturbation_adjust_baseline_invalid():
    inputs = torch.randn(4, 3, 10)

    # Test with invalid baseline type
    with pytest.raises(TypeError):
        LinearInterpolationPerturbation.adjust_baseline("invalid", inputs)

    # Test with mismatched tensor shape
    baseline_tensor = torch.randn(2, 10)
    with pytest.raises(ValueError):
        LinearInterpolationPerturbation.adjust_baseline(baseline_tensor, inputs)

    # Test with mismatched tensor dtype
    baseline_tensor = torch.randn(3, 10, dtype=torch.float64)
    with pytest.raises(ValueError):
        LinearInterpolationPerturbation.adjust_baseline(baseline_tensor, inputs)


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def model():
    return BertModel.from_pretrained("bert-base-uncased").get_input_embeddings()


def test_word_occlusion_perturbator_perturb_string(tokenizer, model):
    perturbator = WordOcclusionPerturbator(tokenizer, model)
    inputs = "This is a test sentence."
    embeddings, mask = perturbator.perturb(inputs)

    n_words = len(inputs.split())

    assert embeddings.shape[0] == n_words
    assert mask.shape == (n_words, n_words)


def test_token_occlusion_perturbator_perturb_string(tokenizer, model):
    perturbator = TokenOcclusionPerturbator(tokenizer, model)
    inputs = "This is another test sentence."
    embeddings, mask = perturbator.perturb(inputs)

    n_tokens = len(tokenizer.tokenize(inputs))
    assert embeddings.shape[0] == n_tokens
    assert mask.shape == (n_tokens, n_tokens)


def test_token_occlusion_perturbator_perturb_iterable(tokenizer, model):
    perturbator = TokenOcclusionPerturbator(tokenizer, model)
    inputs = ["This is a test sentence.", "Another test sentence."]
    perturbed = perturbator.perturb(inputs)

    assert len(perturbed) == len(inputs)
    for (embeddings, mask), sentence in zip(perturbed, inputs, strict=False):
        tokens = tokenizer.tokenize(sentence)
        assert embeddings.shape[0] == len(tokens)
        assert mask.shape == (len(tokens), len(tokens))
