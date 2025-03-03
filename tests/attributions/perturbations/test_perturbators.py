import pytest
import torch
from transformers import BertModel, BertTokenizer

from interpreto.attributions.perturbations.base import GaussianNoisePerturbator
from interpreto.attributions.perturbations.linear_interpolation_perturbation import (
    LinearInterpolationPerturbation,
)
from interpreto.attributions.perturbations.occlusion import TokenOcclusionPerturbator, WordOcclusionPerturbator


def test_gaussian_noise_perturbator_perturb():
    torch.manual_seed(42)
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
def inputs_embedding():
    return BertModel.from_pretrained("bert-base-uncased").get_input_embeddings()


@pytest.fixture
def sentences():
    return [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    ]


def test_tokens_perturbators(tokenizer, inputs_embedding, sentences):
    token_perturbators = [
        TokenOcclusionPerturbator(tokenizer=tokenizer, inputs_embeddings=inputs_embedding),
    ]

    def assert_mask_pertinence(reference, perturbations, mask):
        # Check that a masks corresponds to the perturbation
        # TODO : generalize this function to various shapes and use cases and put it in test utils for other tests
        emb_size = perturbations.shape[-1]
        diff = torch.ne(reference, perturbations).sum(axis=-1) / emb_size
        binary_mask = mask.ne(0)
        assert torch.equal(binary_mask, diff), "Mask is incoherent with applied perturbation"

    sentences_embeddings = [
        inputs_embedding(torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))) for sent in sentences
    ]

    for perturbator in token_perturbators:
        # Test single sentence
        clean_sentence_embeddings = sentences_embeddings[0]

        # perturb the sentence
        perturbed_sentences_embeddings, mask = perturbator.perturb(sentences[0])
        assert perturbed_sentences_embeddings.shape[0] == 1, "Perturbation should have a batch dimension"

        # Check that the mask fits with the perturbation
        assert_mask_pertinence(clean_sentence_embeddings, perturbed_sentences_embeddings.squeeze(0), mask)

        # Test batch of sentences
        for index, (emb, mask) in enumerate(perturbator.perturb(sentences)):
            assert_mask_pertinence(sentences_embeddings[index], emb.squeeze(0), mask)


def test_word_perturbators(tokenizer, inputs_embedding, sentences):
    word_perturbators = [
        WordOcclusionPerturbator(tokenizer=tokenizer, inputs_embeddings=inputs_embedding),
    ]
    for perturbator in word_perturbators:  # noqa: B007
        ...
    # TODO : define tests for word perturbators


def test_occlusion_perturbators(tokenizer, inputs_embedding, sentences):
    occlusion_perturbators = [
        TokenOcclusionPerturbator(tokenizer=tokenizer, inputs_embeddings=inputs_embedding),
        WordOcclusionPerturbator(tokenizer=tokenizer, inputs_embeddings=inputs_embedding),
    ]

    for perturbator in occlusion_perturbators:
        sentence = sentences[0]

        pert, mask = perturbator.perturb(sentence)

        # Check that the mask is identity
        assert torch.all(mask == torch.eye(mask.shape[0]))


def test_word_occlusion_perturbator(tokenizer, inputs_embedding, sentences):
    perturbator = WordOcclusionPerturbator(tokenizer, inputs_embedding)
    embeddings, mask = perturbator.perturb(sentences[0])
    n_words = len(sentences[0].split())
    assert embeddings.shape[1] == n_words
    assert mask.shape == (n_words, n_words)


def test_token_occlusion_perturbator(tokenizer, inputs_embedding, sentences):
    perturbator = TokenOcclusionPerturbator(tokenizer, inputs_embedding)
    embeddings, mask = perturbator.perturb(sentences[0])
    n_tokens = len(tokenizer.tokenize(sentences[0]))
    assert embeddings.shape[1] == embeddings.shape[2] == n_tokens
    assert mask.shape == (n_tokens, n_tokens)
