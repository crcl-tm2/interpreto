"""
Aggregations used at the end of an attribution method
"""
from __future__ import annotations
from abc import ABC
from typing import Any, Collection, Mapping
from interpreto.typing import TokenEmbedding


class Aggregator(ABC):
    """
    Abstract class for aggregation made at the end of attribution methods
    """
    def aggregate(self, results: Collection[Any], mask, **kwargs) -> Any:
        """
        Get results from multiple "Inference wrappers", aggregate results and gives an explaination
        """

    def __call__(self, results: Mapping[TokenEmbedding, Any], mask:Any,**kwargs):
        return self.aggregate(results, mask, **kwargs)
