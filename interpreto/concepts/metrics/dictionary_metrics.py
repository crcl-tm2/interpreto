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
from itertools import combinations

import torch
from jaxtyping import Float
from scipy.optimize import linear_sum_assignment

from interpreto.commons.distances import DistanceFunctionProtocol, cosine_similarity_matrix
from interpreto.concepts.base import ConceptAutoEncoderExplainer


def _cosine_hungarian_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the cosine hungarian distance between two sets of vectors.

    The distance is 0.0 if the two sets of vectors are identical up to a permutation.
    The distance is 1.0 if the two sets of vectors are orthogonal.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        dim (int | tuple[int, ...] | None, optional): Dimensions to reduce. Defaults to None.

    Returns:
        torch.Tensor: The cosine hungarian distance.

    Raises:
        ValueError: If tensor shapes don't match.
        ValueError: If tensor shapes don't match or tensors aren't 2D.
    """
    # cosine distance matrix
    cost_matrix = 1 - cosine_similarity_matrix(x1, x2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    loss = cost_matrix[row_ind, col_ind].sum()

    return loss / x1.shape[0]


def _cosine_max_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the mean max cosine similarity between two tensors.

    Args:
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The mean max cosine similarity.

    Raises:
        ValueError: If tensor shapes don't match.
    """
    cosine_matrix = cosine_similarity_matrix(x1, x2)
    return 1 - cosine_matrix.max(dim=1).values.mean()


class ConceptMatchingAlgorithm(Enum):
    """Code [:octicons-mark-github-24: `concepts/metrics/complexity/stability.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/complexity/stability.py)

    Algorithm used to match concepts between dictionaries.

    Possibilities are:

    - COSINE_HUNGARIAN: the matching is done using the Hungarian loss. TODO: add paper

    - COSINE_MAXIMUM: the matching is done using the maximum cosine similarity.
    """

    COSINE_HUNGARIAN = staticmethod(_cosine_hungarian_distance)
    COSINE_MAXIMUM = staticmethod(
        _cosine_max_distance
    )  # TODO: create a class with this forced and expecting a single dictionary (use at least by the Matryoshka paper)


class Stability:
    """Code [:octicons-mark-github-24: `concepts/metrics/complexity/stability.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/complexity/stability.py)

    Stability metric between sets of dictionaries.

    - If only one dictionary is provided, the metric is a self comparison of the dictionary.
    - If two dictionaries are provided, the metric is a comparison between the two dictionaries.
    - If more than two dictionaries are provided, the metric is the mean of the pairwise comparisons.

    Args:
        concept_explainers (ConceptAutoEncoderExplainer | Float[torch.Tensor, "cpt d"]): The `ConceptAutoEncoderExplainer`s to compare.
        matching_algorithm (DistanceFunctionProtocol, optional): The algorithm used to match concepts between dictionaries. Defaults to ConceptMatchingAlgorithm.COSINE_HUNGARIAN.

    Raises:
        ValueError: If no `ConceptAutoEncoderExplainer`s or dictionary are provided.
        ValueError: If the matching algorithm is not supported.
        ValueError: If the dictionaries are not torch.Tensor.
        ValueError: If the dictionaries have different shapes.
    """

    def __init__(
        self,
        *concept_explainers: ConceptAutoEncoderExplainer | Float[torch.Tensor, "cpt d"],
        matching_algorithm: DistanceFunctionProtocol = ConceptMatchingAlgorithm.COSINE_HUNGARIAN,
    ):
        if len(concept_explainers) < 1:
            raise ValueError("At least one `ConceptAutoEncoderExplainer`s or `torch.Tensor`s must be provided.")

        # if only one explainer is provided, duplicate it for self comparison
        if len(concept_explainers) == 1:
            concept_explainers = concept_explainers * 2

        # extract dictionaries from concept explainers
        self.dictionaries: list[Float[torch.Tensor, "cpt d"]] = [
            ce.get_dictionary() if isinstance(ce, ConceptAutoEncoderExplainer) else ce for ce in concept_explainers
        ]

        expected_shape = None
        for i, dictionary in enumerate(self.dictionaries):
            if not isinstance(dictionary, torch.Tensor):
                raise ValueError(
                    f"Dictionary {i} or dictionary extracted from concept explainer {i} is not a torch.Tensor."
                )

            if len(dictionary.shape) != 2:
                raise ValueError(
                    f"Dictionary {i} or dictionary extracted from concept explainer {i} is not a 2D tensor."
                )

            expected_shape = dictionary.shape if expected_shape is None else expected_shape
            if dictionary.shape != expected_shape:
                raise ValueError(
                    f"Dictionary {i} or dictionary extracted from concept explainer {i} has a different shape from the first dictionary."
                    f"Expected shape: {expected_shape}, got shape: {dictionary.shape}."
                )

        self.distance_function = matching_algorithm

    def compute(self) -> float:
        """Compute the mean score over pairwise comparison scores between dictionaries.

        Returns:
            float: The comparison score.
        """
        comparisons = []
        for dict_1, dict_2 in combinations(self.dictionaries, 2):
            # compute pairwise comparison
            comparison = 1 - self.distance_function(dict_1, dict_2)
            comparisons.append(comparison)

        return torch.stack(comparisons).mean().item()
