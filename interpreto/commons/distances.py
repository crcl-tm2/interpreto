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

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class DistanceFunctionProtocol(Protocol):
    """Protocol representing a callable distance function between two tensors."""

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute distance between two tensors.

        Args:
            x1 (torch.Tensor): The first tensor.
            x2 (torch.Tensor): The second tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The computed distance.
        """
        ...


def wasserstein_1d_distance(x1: torch.Tensor, x2: torch.Tensor, _: Any = None) -> torch.Tensor:
    """Compute the 1D Wasserstein (earth mover's) distance between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        _ (Any, optional): Unused parameter for compatibility.

    Returns:
        torch.Tensor: The 1D Wasserstein distance.

    Raises:
        ValueError: If tensor shapes don't match.
        ValueError: If tensor shapes don't match or tensors aren't 2D.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")
    if len(x1.shape) != 2:
        raise ValueError(f"x1 and x2 must be 2D tensors. Got {len(x1.shape)}D tensors.")

    sorted_x1, _ = x1.sort(dim=0)
    sorted_x2, _ = x2.sort(dim=0)
    return (sorted_x1 - sorted_x2).abs().mean()


def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute the Euclidean (L2) distance between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        dim (int | tuple[int, ...] | None, optional): Dimensions to reduce. Defaults to None.

    Returns:
        torch.Tensor: The Euclidean distance.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")
    return (x1 - x2).norm(p=2, dim=dim)


def average_euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the average Euclidean distance between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The average Euclidean distance.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")

    dim = tuple(range(1, len(x1.shape)))
    sample_wise_euclidean_distance = euclidean_distance(x1, x2, dim=dim)
    return sample_wise_euclidean_distance.mean()


def lp_distance(
    x1: torch.Tensor, x2: torch.Tensor, dim: int | tuple[int, ...] | None = None, *, p: float = 2.0
) -> torch.Tensor:
    """Compute the Lp distance between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        dim (int | tuple[int, ...] | None, optional): Dimensions to reduce. Defaults to None.
        p (float, optional): The power value for the Lp norm. Defaults to 2.0.

    Returns:
        torch.Tensor: The Lp distance.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")
    return (x1 - x2).norm(p=p, dim=dim)


def average_lp_distance(x1: torch.Tensor, x2: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Compute the average Lp distance between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        p (int, optional): The p-norm to use. Defaults to 2.

    Returns:
        torch.Tensor: The average Lp distance.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")

    dim = tuple(range(1, len(x1.shape)))
    sample_wise_lp_distance = lp_distance(x1, x2, p=p, dim=dim)
    return sample_wise_lp_distance.mean()


def kl_divergence(x1: torch.Tensor, x2: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute the KL divergence between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        dim (int | tuple[int, ...] | None, optional): Dimensions to reduce. Defaults to None.

    Returns:
        torch.Tensor: The KL divergence.

    Raises:
        ValueError: If tensor shapes don't match.
        ValueError: If tensor values are outside the [0, 1] range.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")

    if any(x1 < 0) or any(x2 < 0) or any(x1 > 1) or any(x2 > 1):
        raise ValueError(
            "KL divergence is undefined for negative values or values > 1.",
            "The KL divergence is only defined on probability distributions.",
        )

    return (x1 * (x1 / x2).log()).sum(dim=dim)


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute the cosine similarity between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The cosine similarity.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Tensor shapes must match. Got {x1.shape} and {x2.shape}.")

    x1_norm = x1.norm(p=2, dim=dim)
    x2_norm = x2.norm(p=2, dim=dim)

    return (x1 * x2).sum(dim=dim) / (x1_norm * x2_norm)


def cosine_similarity_matrix(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the cosine similarity matrix between two sets of vectors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The cosine similarity matrix.

    Raises:
        ValueError: If tensor shapes don't match.
        ValueError: If tensor shapes don't match or tensors aren't 2D.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shapes of x1 and x2 must match. Got {x1.shape}, {x2.shape}.")

    if len(x1.shape) != 2:
        raise ValueError(f"x1 and x2 must be 2D tensors. Got {len(x1.shape)}D tensors.")

    x1_normed = x1 / x1.norm(p=2, dim=1, keepdim=True)
    x2_normed = x2 / x2.norm(p=2, dim=1, keepdim=True)

    return x1_normed @ x2_normed.T


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute the cosine distance between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        dim (int | tuple[int, ...] | None, optional): Dimensions to reduce. Defaults to None.

    Returns:
        torch.Tensor: The cosine distance.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    return 1 - cosine_similarity(x1, x2, dim=dim)


class DistanceFunctions(Enum):
    """Enum of callable functions for computing distances between tensors.

    Members:
        WASSERSTEIN_1D: Computes the 1D Wasserstein (earth mover's) distance between two tensors.
        EUCLIDEAN: Computes the Euclidean (L2) distance between two tensors.
        AVERAGE_EUCLIDEAN: Computes the average Euclidean distance between two tensors of samples.
        LP: Computes the Lp distance (generalization of Euclidean) between two tensors.
        AVERAGE_LP: Computes the average Lp distance between two tensors of samples.
        KL: Computes the Kullback-Leibler divergence between two tensors.
    """

    WASSERSTEIN_1D = staticmethod(wasserstein_1d_distance)
    EUCLIDEAN = staticmethod(euclidean_distance)
    AVERAGE_EUCLIDEAN = staticmethod(average_euclidean_distance)
    LP = staticmethod(lp_distance)
    AVERAGE_LP = staticmethod(average_lp_distance)
    KL = staticmethod(kl_divergence)
    COSINE = staticmethod(cosine_distance)
