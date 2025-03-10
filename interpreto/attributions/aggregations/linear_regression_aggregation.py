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
Aggregator for LIME and KernelSHAP
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch.nn import functional as F

from interpreto.attributions.aggregations.base import Aggregator


def hamming_distance(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hamming distance between a mask and the original input
    """
    return torch.sum(mask, dim=1).float()


def euclidean_distance(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance between a mask and the original input
    """
    return torch.norm(mask, dim=1)


def cosine_distance(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine distance between a mask and the original input
    """
    return F.cosine_similarity(torch.ones_like(mask), mask, dim=1)


class DistanceFunctions(Enum):
    """
    Enumeration of available distance functions.
    """

    HAMMING = hamming_distance
    EUCLIDEAN = euclidean_distance
    COSINE = cosine_distance


def exponential_kernel(distances: torch.Tensor, kernel_width: float) -> torch.Tensor:
    """
    Compute the exponential kernel for a given distance matrix
    """
    return torch.exp(-(distances**2) / (kernel_width**2))


def ones_kernel(distances: torch.Tensor, kernel_width: float) -> torch.Tensor:
    """
    Compute the ones kernel for a given distance matrix
    """
    return torch.ones_like(distances)


class Kernels(Enum):
    """
    Enumeration of available kernels.
    """

    EXPONENTIAL = exponential_kernel
    ONES = ones_kernel


def default_kernel_width_fn(mask: torch.Tensor) -> float:
    """
    Compute the default kernel width for a given mask
    """
    return (mask.shape[1] ** 0.5) * 0.75  # kernel parameter inspired by LIME


class LinearRegressionAggregator(Aggregator):
    """
    Aggregator for masked perturbations using a linear model
    """

    def __init__(
        self,
        distance_function: Callable | None = None,
        similarity_kernel: Callable = exponential_kernel,
        kernel_width: float | Callable | None = default_kernel_width_fn,
    ):
        self.distance_function = distance_function
        self.similarity_kernel = similarity_kernel
        self.kernel_width = kernel_width

    @jaxtyped
    @beartype
    def aggregate(
        self, results: Float[torch.Tensor, "p"], mask: Float[torch.Tensor, "p", "l"]
    ) -> Float[torch.Tensor, "l"]:
        if self.distance_function is not None:  # LIME
            # Compute distance between perturbations and original input using the mask
            distances = self.distance_function(mask)  # (p,)

            # Compute the similarities between perturbations and original input using the distances
            if isinstance(self.kernel_width, Callable):
                kernel_width = self.kernel_width(mask)
            else:
                assert isinstance(self.kernel_width, float)
                kernel_width = self.kernel_width
            weights = self.similarity_kernel(distances, kernel_width)  # (p,)
        else:  # Kernel SHAP
            weights = self.similarity_kernel(mask[:, 0], None)  # (p,)

        # Compute closed form solution for the linear model
        # Add a constant column for the bias term
        X = torch.cat([torch.ones(results.shape[0], 1), mask], dim=1)  # shape: (p, l + 1)

        # \theta =(X^T W X)^{-1} (X^T W y)
        XT_W = results.T * weights  # shape: (l + 1, p)
        theta = torch.inverse(XT_W @ X) @ (XT_W @ results)  # shape: (l + 1,)

        # exclude the bias term
        token_importance = theta[1:]  # shape: (l,)

        return token_importance
