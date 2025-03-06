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

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import Perturbator


class SobolPerturbator(Perturbator):
    def __init__(self, tokenizer: PreTrainedTokenizer, baseline="[MASK]", n_perturbations=1000, proba=0.5):
        """
        - tokenizer: Hugging Face tokenizer associated with the model
        - baseline: replacement token (e.g. “[MASK]”)
        - n_perturbations: number of Monte Carlo samples
        - proba: probability of keeping a token (i.e. putting 1 in the mask)
        """
        self.tokenizer = tokenizer
        self.baseline = baseline
        self.n_perturbations = n_perturbations
        self.proba = proba

    def perturb(self, text, Sobol_indices="first order"):
        """
        Generates perturbations for the entire input and for each "real" token position only.

        Parameters:
          - text : original sentence (str)
          - Sobol_indices : "first order" or "total" (str)

        Returns a dictionary containing:
          - "origin perturbated inputs": dictionary with input_ids and attention_mask for the full-sequence perturbations.
          - "list of perturbated inputs for each token": a dict mapping each real token's position to its own perturbation inputs.
          - "real_tokens": a dict mapping each real token position to its token string.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Tokenize text with offsets (includes special tokens)
        inputs_model = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = (
            inputs_model["offset_mapping"].squeeze(0).tolist()
        )  # List of (start, end) pairs for each token
        input_ids = inputs_model["input_ids"].squeeze(0)  # Shape: (seq_len,)
        attention_mask = inputs_model["attention_mask"].squeeze(0)  # Shape: (seq_len,)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Identify "real" tokens: those with a nonzero span in the offset mapping.
        real_indices = [i for i, (start, end) in enumerate(offset_mapping) if (end - start) > 0]
        real_tokens = {i: all_tokens[i] for i in real_indices}

        baseline_id = self.tokenizer.convert_tokens_to_ids(self.baseline)
        seq_len = input_ids.shape[0]
        print(
            "Number of all tokens:",
            len(all_tokens),
            "number of real tokens:",
            len(real_tokens),
            "Sequence length:",
            seq_len,
        )
        print("All tokens:", all_tokens)
        print("Real tokens:", real_tokens)

        # Create origin perturbations for the entire sequence.
        origin_masks = []
        origin_input_ids_list = []
        origin_attention_mask_list = []
        for _ in range(self.n_perturbations):
            # Create a binary mask with probability self.proba for keeping the token.
            origin_mask = torch.bernoulli(torch.full((seq_len,), self.proba)).long()
            origin_input_ids = input_ids * origin_mask + baseline_id * (1 - origin_mask)
            origin_masks.append(origin_mask)
            origin_input_ids_list.append(origin_input_ids)
            origin_attention_mask_list.append(attention_mask)
        origin_input_ids_tensor = torch.stack(origin_input_ids_list).to(device)
        origin_attention_mask_tensor = torch.stack(origin_attention_mask_list).to(device)
        origin_inputs_model = {
            "input_ids": origin_input_ids_tensor,
            "attention_mask": origin_attention_mask_tensor,
        }

        # For each real token position, create perturbations by flipping that token's mask.
        pert_inputs_model_per_token = {}
        for i in real_indices:
            pert_input_ids_list = []
            pert_attention_mask_list = []
            for j in range(self.n_perturbations):
                pert_mask = origin_masks[j].clone()
                # Flip the bit at token position i.
                pert_mask[i] = 1 - pert_mask[i]
                # If computing total Sobol indices, flip all bits except the i-th bit.
                if Sobol_indices == "total":
                    pert_mask = 1 - pert_mask
                pert_input_ids = input_ids * pert_mask + baseline_id * (1 - pert_mask)
                pert_input_ids_list.append(pert_input_ids)
                pert_attention_mask_list.append(attention_mask)
            pert_inputs_model = {
                "input_ids": torch.stack(pert_input_ids_list).to(device),
                "attention_mask": torch.stack(pert_attention_mask_list).to(device),
            }
            pert_inputs_model_per_token[i] = pert_inputs_model

        return {
            "origin perturbated inputs": origin_inputs_model,
            "list of perturbated inputs for each token": pert_inputs_model_per_token,
            "real_tokens": real_tokens,
        }
