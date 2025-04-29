from __future__ import annotations

import torch

from interpreto.attributions.perturbations.base import TokenMaskBasedPerturbator
from interpreto.commons.granularity import GranularityLevel


class OcclusionPerturbator(TokenMaskBasedPerturbator):
    """
    Basic class for occlusion perturbations
    """

    __slots__ = ()

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
        replace_token_id: int = 0,
    ):
        super().__init__(
            replace_token_id=replace_token_id,
            inputs_embedder=inputs_embedder,
            n_perturbations=-1,
            granularity_level=granularity_level,
        )

    def get_mask(self, mask_dim: int) -> torch.Tensor:
        return torch.cat([torch.zeros(1, mask_dim), torch.eye(mask_dim)], dim=0)
