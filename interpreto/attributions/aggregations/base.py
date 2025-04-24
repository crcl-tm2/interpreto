# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Aggregations used at the end of an attribution method
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class Aggregator:
    """
    Abstract class for aggregation made at the end of attribution methods
    """

    def aggregate(self, results: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Get results from multiple "Inference wrappers", aggregate results and gives an explanation
        """
        return results

    def __call__(self, results: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.aggregate(results, mask)


class TorchAggregator(Aggregator):
    """
    Basic aggregator using built-in torch methods to perform aggregation
    """

    _method: Callable[[torch.Tensor, int], torch.Tensor]

    def aggregate(self, results: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # TODO: check dimension with explicit jax typing for results parameter
        return self._method(results, dim=0)


class MeanAggregator(TorchAggregator):
    """
    Mean of attributions
    """

    _method = torch.mean


class SquaredMeanAggregator(Aggregator):
    """
    Mean of squares of attributions
    """

    @staticmethod
    def _method(results: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.mean(torch.square(results), dim=dim)


class SumAggregator(TorchAggregator):
    """
    Sum of attributions
    """

    _method = torch.sum


class VarianceAggregator(TorchAggregator):
    """
    Variance of attributions
    """

    _method = torch.var


class MaskwiseMeanAggregator(Aggregator):
    """
    TODO : add docstring
    """

    def aggregate(self, results: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.einsum("pt,pl->tl", results, mask) / mask.sum(dim=1)


class OcclusionAggregator(Aggregator):
    """
    TODO : add docstring
    """

    def aggregate(self, results: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # first prediction is reference, unmodified input
        scores = results[..., 0, :] - results[..., 1:, :]
        mask = mask[..., 1:, :]
        return torch.einsum("pt,pl->tl", scores, mask) / mask.sum(dim=1)
