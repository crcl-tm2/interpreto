import pytest
import torch

from interpreto.attributions.perturbations import SobolTokenPerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers, SobolIndicesOrders


@pytest.mark.parametrize(
    "sobol_indices_order, sampler",
    [
        (SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.SOBOL),
        (SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.SOBOL),
        (SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.HALTON),
        (SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.HALTON),
        (SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.LatinHypercube),
        (SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.LatinHypercube),
    ],
)
def test_get_masks(sobol_indices_order, sampler):
    k = 10
    perturbator = SobolTokenPerturbator(
        inputs_embedder=None,
        n_token_perturbations=k,
        sobol_indices_order=sobol_indices_order,
        sampler=sampler,
    )

    sizes = list(range(2, 20, 3))

    masks = [perturbator.get_mask(size) for size in sizes]

    for l, mask in zip(sizes, masks, strict=True):
        assert mask.shape == ((l + 1) * k, l)
        initial_mask = mask[:k]

        # verify token-wise mask compared to the initial mask
        for i in range(l):
            token_mask_i = mask[(i + 1) * k : (i + 2) * k]
            if sobol_indices_order == SobolIndicesOrders.FIRST_ORDER:
                assert torch.all(torch.isclose(token_mask_i[:, i], 1 - initial_mask[:, i], atol=1e-5))
                if i != 0:
                    assert torch.all(torch.isclose(token_mask_i[:, :i], initial_mask[:, :i], atol=1e-5))
                if i != l - 1:
                    assert torch.all(torch.isclose(token_mask_i[:, i + 1 :], initial_mask[:, i + 1 :], atol=1e-5))
            else:
                assert torch.all(torch.isclose(token_mask_i[:, i], initial_mask[:, i], atol=1e-5))
                if i != 0:
                    assert torch.all(torch.isclose(token_mask_i[:, :i], 1 - initial_mask[:, :i], atol=1e-5))
                if i != l - 1:
                    assert torch.all(torch.isclose(token_mask_i[:, i + 1 :], 1 - initial_mask[:, i + 1 :], atol=1e-5))


test_get_masks(SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.LatinHypercube)
