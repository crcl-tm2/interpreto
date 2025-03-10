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
Sobol perturbations for NLP
"""

from __future__ import annotations

from enum import Enum

import torch
from scipy.stats import qmc
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import TokenMaskBasedPerturbator


class SobolIndicesOrders(Enum):
    """
    Enumeration of available Sobol indices orders.
    """

    FIRST_ORDER = "first order"
    TOTAL_ORDER = "total order"


class SequenceSamplers(Enum):
    """
    Enumeration of available samplers for Sobol perturbations.
    """

    SOBOL = qmc.Sobol
    HALTON = qmc.Halton
    LatinHypercube = qmc.LatinHypercube


class SobolTokenPerturbator(TokenMaskBasedPerturbator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        inputs_embedder: torch.nn.Module | None = None,
        n_token_perturbations: int = 30,
        granularity_level: str = "token",
        baseline: str = "[MASK]",
        sobol_indices_order: SobolIndicesOrders = SobolIndicesOrders.FIRST_ORDER,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        device: torch.device | None = None,
    ):
        """
        Initialize the perturbator.

        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            inputs_embedder (torch.nn.Module | None): optional inputs embedder
            nb_token_perturbations (int): number of Monte Carlo samples perturbations for each token.
            granularity_level (str): granularity level of the perturbations (token, word, sentence, etc.)
            baseline (str): replacement token (e.g. “[MASK]”)
            sobol_indices (SobolIndicesOrders): Sobol indices order, either `FIRST_ORDER` or `TOTAL_ORDER`.
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
            device (torch.device): device on which the perturbator will be run
        """
        super().__init__(
            tokenizer=tokenizer,
            inputs_embedder=inputs_embedder,
            n_perturbations=None,
            mask_token=baseline,
            granularity_level=granularity_level,
            device=device,
        )
        self.tokenizer = tokenizer
        self.n_token_perturbations = n_token_perturbations
        self.sobol_indices_order = sobol_indices_order
        self.sampler = sampler

    def get_single_input_mask(self, l: int):
        """
        Generates a binary mask for each token in the sequence.

        Args:
            l (int): The length of the sequence.

        Returns:
            masks (torch.Tensor): A tensor of shape ((l + 1) * k, l).
        """
        k = self.n_token_perturbations
        # Initial random mask. Shape:(k, l)
        initial_mask = torch.Tensor(self.sampler(l).random(k), device=self.device)

        # Expand mask across all perturbation steps. Shape ((l + 1) * k, l)
        mask = initial_mask.repeat((l + 1, 1))

        # Generate index tensor for perturbations. Shape: (l * k)
        col_indices = torch.arange(l, device=self.device).repeat_interleave(k)

        # Compute the start and end indices. Shape: (l * k)
        row_indices = torch.arange(k * l, device=self.device) + k

        # Flip the selected mask values without a loop
        mask[row_indices, col_indices] = 1 - mask[row_indices, col_indices]

        if self.sobol_indices_order == SobolIndicesOrders.TOTAL_ORDER:
            mask[k:] = 1 - mask[k:]

        return mask

    def get_mask(self, sizes: tuple | list[int]) -> list[torch.Tensor]:
        """
        Generates a binary mask for each token in the sequence.

        Args:
            sizes (list[int]): A list of sequence lengths (l).

        Returns:
            masks_list (list[torch.Tensor]): List of tensors of shape ((l + 1) * k, l).
        """
        if isinstance(sizes, int):
            return [self.get_single_input_mask(sizes)]

        if isinstance(sizes, list):
            return [self.get_single_input_mask(size) for size in sizes]

        if isinstance(sizes, tuple):
            return torch.repeat(self.get_single_input_mask(sizes[0]).unsqueeze(0), sizes[1], dim=0)
