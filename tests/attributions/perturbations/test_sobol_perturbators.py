import pytest

from interpreto.attributions.perturbations import SobolTokenPerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers, SobolIndicesOrders


@pytest.mark.parametrize(
    "sobol_indices, sampler",
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
        tokenizer=None,
        inputs_embedder=None,
        nb_token_perturbations=k,
        granularity_level="token",
        baseline="[MASK]",
        sobol_indices_order=sobol_indices_order,
        sampler=sampler,
    )

    sizes = list(range(2, 20, 3))

    masks = perturbator.get_masks(sizes)

    for l, mask in zip(sizes, masks, strict=True):
        assert mask.shape == ((l + 1) * k, k)
        initial_mask = mask[:k]

        # verify token-wise mask compared to the initial mask
        for i in range(l):
            token_mask_i = mask[(i + 1) * k : (i + 2) * k]
            if sobol_indices_order == SobolIndicesOrders.FIRST_ORDER:
                assert all(token_mask_i[i] != initial_mask[i])
                assert all(token_mask_i[:i] == initial_mask[:i])
                if i != l - 1:
                    assert all(token_mask_i[i + 1 :] != initial_mask[i + 1 :])
            else:
                assert all(token_mask_i[i] == initial_mask[i])
                assert all(token_mask_i[:i] != initial_mask[:i])
                if i != l - 1:
                    assert all(token_mask_i[i + 1 :] == initial_mask[i + 1 :])
