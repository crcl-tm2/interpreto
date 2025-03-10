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
Random perturbation for token-wise masking, used in LIME
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import GranularityLevel, TokenMaskBasedPerturbator


class RandomMaskedTokenPerturbator(TokenMaskBasedPerturbator):
    """
    Perturbator adding random masking to the input tensor
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        n_perturbations: int = 1,
        mask_token: str = None,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
    ):
        super().__init__(
            tokenizer=tokenizer,
            inputs_embedder=inputs_embedder,
            n_perturbations=n_perturbations,
            mask_token=mask_token,
            granularity_level=granularity_level,
        )

    def get_mask(self, model_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Method returning a random perturbation mask for a given set of inputs.

        The created mask should be of size (batch_size, n_perturbations, mask_dimension)
        where mask_dimension is the length of the sequence according to the granularity level (number of tokens, number of words, number of sequences...)

        Args:
            model_inputs (Mapping[str, torch.Tensor]): mapping given by the tokenizer

        Returns:
            torch.Tensor: mask to apply
        """
        # Example implementation that returns a no-perturbation mask
        # TODO factorize the getting of the mask dimension to outside of `get_mask`
        mask_dimension = (
            GranularityLevel.get_association_matrix(model_inputs, self.granularity_level)
            .sum(dim=(-1, -2))
            .max()
            .int()
            .item()
        )
        batch_size = model_inputs["input_ids"].shape[0]
        return torch.rand((batch_size, self.n_perturbations, mask_dimension))
