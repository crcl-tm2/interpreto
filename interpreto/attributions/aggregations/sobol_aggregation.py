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

import torch

from interpreto.attributions.aggregations.base import Aggregator


class SobolAggregator(Aggregator):
    """
    Aggregates Sobol indices from model outputs.
    """

    def __init__(self, n_token_perturbations: int):
        self.n_token_perturbations = n_token_perturbations

    def single_input_aggregate(self, scores: torch.Tensor, _) -> torch.Tensor:
        """
        Compute the Sobol indices from the model outputs perturbed inputs.

        Args:
            scores (torch.Tensor): The model outputs on perturbed inputs. Shape: (p,) with p = (l + 1) * k

        Returns:
            token_importance (torch.Tensor): The Sobol attribution indices for each token. Shape: (l,)
        """
        k = self.n_token_perturbations

        # Compute token-wise variance on the initial mask
        initial_scores = scores[:k]  # Shape: (k,)
        initial_var = torch.var(initial_scores)  # Shape: (1,)

        # Compute token-wise sobol attribution indices
        token_scores = scores[k:].view(-1, k)  # Shape: (l, k)
        difference = token_scores - initial_scores.unsqueeze(1)  # Shape: (l, k)
        token_importance = torch.mean(difference**2, dim=1) / (initial_var + 1e-6)  # Shape: (l,)
        return token_importance

    def aggregate(self, scores_list: list(torch.Tensor), _) -> list(torch.Tensor):
        """
        Compute the Sobol indices from the model outputs perturbed inputs.

        Args:
            scores_list (list(torch.Tensor)): The list of model outputs on perturbed inputs. Elements shape: (p,) with p = (l + 1) * k

        Returns:
            list(torch.Tensor): The Sobol attribution indices for each token. Shape: (l,)
        """
        return [self.single_input_aggregate(scores, _) for scores in scores_list]
