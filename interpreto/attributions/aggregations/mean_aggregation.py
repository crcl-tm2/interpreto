from __future__ import annotations

from typing import Any

import torch

from interpreto.attributions.aggregations.base import Aggregator


class MeanAggregation(Aggregator):
    """
    Mean aggregation of attributions
    """

    def aggregate(self, results: torch.Tensor, _, **kwargs) -> Any:
        return results.mean(dim=1)  # TODO: verify dimension
