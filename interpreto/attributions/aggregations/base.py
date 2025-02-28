"""
Aggregations used at the end of an attribution method
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch


class Aggregator(ABC):
    """
    Abstract class for aggregation made at the end of attribution methods
    """

    @abstractmethod
    def aggregate(self, results: Iterable[Any], mask) -> Any:
        """
        Get results from multiple "Inference wrappers", aggregate results and gives an explanation
        """

    def __call__(self, results: Iterable[Any], mask: Any):
        return self.aggregate(results, mask)

class TorchAggregator(Aggregator):
    """
    Basic aggregator using built-in torch methods to perform aggregation
    """
    __method:str

    def aggregate(self, results: torch.Tensor, _) -> torch.Tensor:
        # TODO: check dimension with explicit jax typing for results parameter
        return self.__method(results, dim=1)

class MeanAggregator(TorchAggregator):
    """
    Mean of attributions
    """
    __method=torch.Tensor.mean


class SquaredMeanAggregator(Aggregator):
    """
    Square of mean of attributions
    """
    # TODO : remake this class with __method as a function chain of torch.Tensor.mean and torch.Tensor.square
    # __method=torch.Tensor.mean
    def aggregate(self, results: torch.Tensor, _) -> Any:
        return torch.Tensor.mean(results, _)**2

class SumAggregator(TorchAggregator):
    """
    Sum of attributions
    """
    __method=torch.Tensor.sum

class VarianceAggregator(TorchAggregator):
    """
    Variance of attributions
    """
    __method=torch.Tensor.var
