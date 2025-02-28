import pytest
import torch

from interpreto.attributions.perturbations.base import GaussianNoisePerturbator
from interpreto.attributions.perturbations.linear_interpolation_perturbation import (
    LinearInterpolationPerturbation,
)


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
