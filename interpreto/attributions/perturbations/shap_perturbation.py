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
Perturbation for SHAP
"""

from __future__ import annotations

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from interpreto.attributions.perturbations.base import TokenMaskBasedPerturbator
from interpreto.commons.granularity import GranularityLevel


class ShapTokenPerturbator(TokenMaskBasedPerturbator):
    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
        replace_token_id: int = 0,
        n_perturbations: int = 1000,
        device: torch.device | None = None,
    ):
        """
        Initialize the perturbator.

        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            inputs_embedder (torch.nn.Module | None): optional inputs embedder
            replace_token_id (int): the token id to use for replacing the masked tokens
            n_perturbations (int): the number of perturbations to generate
            device (torch.device): device on which the perturbator will be run
        """
        super().__init__(
            inputs_embedder=inputs_embedder,
            n_perturbations=n_perturbations,
            replace_token_id=replace_token_id,
            granularity_level=granularity_level,
        )
        self.device = device

    @jaxtyped
    def get_mask(self, mask_dim: int) -> Float[Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Generates a binary mask for each token in the sequence.

        The perturbed instances are sampled that way:
         - We choose a number of selected features k, considering the distribution
                p(k) = (nb_features - 1) / (k * (nb_features - k))
            where nb_features is the total number of features in the interpretable space
         - Then we randomly select a binary vector with k ones, all the possible sample
           are equally likely. It is done by generating a random vector with values drawn
           from a normal distribution and keeping the top k elements which then will be 1
           and other values are 0.
         Since there are nb_features choose k vectors with k ones, this weighted sampling
         is equivalent to applying the Shapley kernel for the sample weight, defined as:
            k(nb_features, k) = (nb_features - 1)/(k*(nb_features - k)*(nb_features choose k))
        This trick is the one used in the Captum library: https://github.com/pytorch/captum

        Args:
            l (int): The length of the sequence.

        Returns:
            masks (torch.Tensor): A tensor of shape ((l + 1) * k, l).
        """
        # Simplify typing
        p, l = self.n_perturbations, mask_dim

        # Generate a random number of selected features k for each perturbation
        possible_k: Float[Tensor, "{l}"] = torch.arange(1, l + 1, dtype=torch.float, device=self.device)
        probability_to_select_k_elements: Float[Tensor, l] = (l - 1) / (possible_k * (l - possible_k))
        k: Float[Tensor, p] = torch.multinomial(probability_to_select_k_elements, p, replacement=True)

        # Generate a random binary mask for each perturbation
        rand_values: Float[Tensor, "{p} {l}"] = torch.rand(p, l, dtype=torch.float, device=self.device)
        thresholds: Float[Tensor, "{p}"] = torch.stack(
            [torch.kthvalue(rand_values[i], int(k[i]), dim=0).values for i in range(p)]
        )

        mask: Float[Tensor, "{p} {l}"] = (rand_values < thresholds.unsqueeze(1)).float()

        return mask
