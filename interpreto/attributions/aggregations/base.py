"""
Aggregations used at the end of an attribution method
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import Any

from interpreto.typing import TokenEmbedding


class Aggregator(ABC):
    """
    Abstract class for aggregation made at the end of attribution methods
    """

    @abstractmethod
    def aggregate(self, results: Collection[Any], mask, **kwargs) -> Any:
        """
        Get results from multiple "Inference wrappers", aggregate results and gives an explanation
        """

    def __call__(self, results: Mapping[TokenEmbedding, Any], mask: Any, **kwargs):
        return self.aggregate(results, mask, **kwargs)
