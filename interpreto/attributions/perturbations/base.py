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
Base classes for perturbations used in attribution methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable

import torch

from interpreto.typing import ModelInput

class GranularityLevel(Enum):
    ALL_TOKENS = "all_tokens"
    TOKEN = "token"
    WORD = "word"

def granularity_association_matrix(tokens_ids,
                                   granularity_level:GranularityLevel=GranularityLevel.TOKEN):
    print(list(tokens_ids.keys()))
    n, l_p = tokens_ids["input_ids"].shape
    if granularity_level == GranularityLevel.ALL_TOKENS:
        return torch.eye(l_p).unsqueeze(0).expand(n, -1, -1)
    if granularity_level == GranularityLevel.TOKEN:
        # TODO : refaire ça ?
        perturbable_mask = tokens_ids["offset_mapping"].sum(dim=-1).bool().long()
        perturbable_matrix = torch.diag_embed(perturbable_mask)
        non_empty_rows_mask = perturbable_matrix.sum(dim=2) != 0
        l_t = non_empty_rows_mask.sum(dim=1)
        result = torch.zeros(n, l_t.max(), l_p)
        for i in range(n):
            result[i, :l_t[i], :]=perturbable_matrix[i, non_empty_rows_mask[i]]
        return result
    raise NotImplementedError(f"Granularity level {granularity_level} not implemented")

class BasePerturbator:
    def __init__(self, tokenizer:PreTrainedTokenizer|None=None,
                                inputs_embedder:torch.nn.Module|None=None):
        self.tokenizer = tokenizer
        self.inputs_embedder = inputs_embedder

    @singledispatchmethod
    def perturb(self, inputs) -> dict[str, torch.Tensor]:
        """
        Method to perturb an input, should return a collection of perturbed elements and their associated masks
        """
        raise NotImplementedError(f"Method perturb not implemented for type {type(inputs)} in {self.__class__.__name__}")

    @perturb.register(str)
    def _(self, inputs:str) -> Mapping[str, torch.Tensor]:
        return self.perturb([inputs])

    @perturb.register(Iterable)
    def _(self, inputs:Iterable[str]) -> Mapping[str, torch.Tensor]:
        perturbed_sentences = self.perturb_strings(inputs)

        if self.tokenizer is None:
            raise ValueError("A tokenizer is required to perturb strings. Please provide a tokenizer when initializing the perturbator or specify it with 'perturbator.tokenizer = some_tokenizer'")
        tokens = self.tokenizer(perturbed_sentences, truncation=True, return_tensors='pt', padding=True, return_offsets_mapping=True)
        return self.perturb(tokens)

    @perturb.register(Mapping)
    def _(self, inputs:Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # TODO : do not perturb special tokens (use offset mapping)
        assert "offset_mapping" in inputs, "Offset mapping is required to perturb tokens, specify the 'return_offsets_mapping=True' parameter when tokenizing the input"

        inputs = self.perturb_ids(inputs)
        if self.inputs_embedder is None:
            return inputs
        try:
            embeddings = self.perturb_tensors(self.inputs_embedder(inputs))
            return {"inputs_embeds":embeddings}# add complementary data in dict
        except NotImplementedError:
            return inputs

    @perturb.register(torch.Tensor)
    def _(self, inputs:torch.Tensor) -> Mapping[str, torch.Tensor]:
        return  {"inputs_embeds": self.perturb_tensors(inputs)}

    def perturb_strings(self, strings:Iterable[str]) -> Iterable[str]:
        # Default implementation, should be overriden
        return strings

    def perturb_ids(self, model_inputs:Mapping) -> Mapping[str, torch.Tensor]:
        # Default implementation, should be overriden
        return model_inputs

    def perturb_tensors(self, tensors:torch.Tensor) -> torch.Tensor:
        # Default implementation, should be overriden
        raise NotImplementedError(f"No way to perturb embeddings has been defined in {self.__class__.__name__}")

class MaskBasedPerturbator(BasePerturbator):
    def get_model_inputs_mask(self, *args, **kwargs):
        # TODO : implementation par defaut
        ...

    @property
    def default_mask_id(self):
        return self.tokenizer.mask_token_id

    def apply_mask(self, inputs:torch.Tensor, mask:torch.Tensor, mask_value:torch.Tensor):
        base = torch.einsum("nld,npl->npld", inputs, 1 - mask)
        masked = torch.einsum("npl,d->npld", mask, mask_value)
        return base + masked

class TokenMaskBasedPerturbator(MaskBasedPerturbator):
    def __init__(self, tokenizer:PreTrainedTokenizer|None=None,
                        inputs_embedder:torch.nn.Module|None=None,
                        n_perturbations:int=1,
                        granularity_level:str="token"):
        super().__init__(tokenizer=tokenizer, inputs_embedder=inputs_embedder)
        self.n_perturbations = n_perturbations
        self.granularity_level = granularity_level

    def get_mask(self, model_inputs:Mapping) -> torch.Tensor:
        # input : 1d tensor containing n values of l_spec where :
        # l_spec = length of the sequence according to certain granularity level (nb words, nb tokens, nb sentences, etc.)
        # n = batch size
        # TODO : redefine this
        return ...#[torch.randbool(p, l).long() for l in sizes]

    def get_t_mask_from_p_mask(self, model_inputs:Mapping, p_mask:torch.Tensor)->torch.Tensor:
        # TODO : eventually store gran matrix in tokens to avoid recomputing it ?
        t_gran_matrix = granularity_association_matrix(model_inputs, self.granularity_level)
        return torch.einsum("npr,ntr->npt", p_mask, t_gran_matrix) / t_gran_matrix.sum(dim=-1)


    def get_p_mask_from_t_mask(self, model_inputs:Mapping, t_mask:torch.Tensor) -> torch.Tensor:
        # mask : T-mask furnished by get_theorical_masks
        # granularity_matrix : for editable tokens to tokens : matrix of editable tokens (offset_mapping.sum(dim=-1).bool().long())
        gran_matrix = granularity_association_matrix(model_inputs, self.granularity_level)
        # TODO : eventually store gran matrix in tokens to avoid recomputing it ?
        return torch.einsum("npt,ntr->npr", t_mask, gran_matrix)

    def get_model_inputs_mask(self, model_inputs:Mapping) -> torch.Tensor:
        return self.get_p_mask_from_t_mask(model_inputs, self.get_mask(model_inputs))

    def perturb_ids(self, model_inputs:Mapping) -> dict[str, torch.Tensor]:
        #tokens["perturbation_mask"] = self.get_mask(tokens) * tokens["offset_mapping"].sum(dim=-1).bool().long().unsqueeze(-1)
        model_inputs["perturbation_mask"] = self.get_model_inputs_mask(model_inputs)

        model_inputs["input_ids"] = self.apply_mask(
            inputs=model_inputs["input_ids"].unsqueeze(-1),
            mask=model_inputs["perturbation_mask"],
            mask_value=torch.Tensor([self.default_mask_id])
        )
        return model_inputs

class EmbeddingsMaskBasedPerturbator(MaskBasedPerturbator):
    def default_mask_vector(self):
        return self.inputs_embedder.weight[self.default_mask_id]

    def perturb_embeddings(self, embeddings):
        mask = ...
        perturbed_embeddings = self.apply_mask(embeddings, mask, self.default_mask_vector)
        return {"perturbation_mask":mask, "inputs_embeds":perturbed_embeddings}

# Change to general occlusion perturbator
class TokenOcclusionPerturbator(TokenMaskBasedPerturbator):
    def __init__(self, tokenizer:PreTrainedTokenizer|None=None,
                        inputs_embedder:torch.nn.Module|None=None):
        super().__init__(tokenizer=tokenizer, inputs_embedder=inputs_embedder)
        self.granularity_level = GranularityLevel.TOKEN

    def get_mask(self, model_inputs:Mapping) -> torch.Tensor:
        # TODO : add sugar to get mask dimensions
        mask_dim = granularity_association_matrix(model_inputs, self.granularity_level).sum().int().item()
        n = model_inputs["input_ids"].shape[0]
        return torch.eye(mask_dim).unsqueeze(0).expand(n, -1, -1)

    # def get_real_mask(self, tokens:Mapping) -> torch.Tensor:
    #     self.granularity_level = GranularityLevel.TOKEN
    #     # TODO : dégager ça
    #     #perturbable_mask = tokens["offset_mapping"].sum(dim=-1).bool().long()
    #     #attention_masks = tokens["offset_mapping"]
    #     #max_attention_mask_length = attention_masks.sum(dim=1).max().item()
    #     return torch.diag_embed(perturbable_mask)[:, :max_attention_mask_length]

class GaussianNoisePerturbator():
    # TODO : remake this with new arch

# class GaussianNoisePerturbator(TensorPerturbator):
#     """
#     Perturbator adding gaussian noise to the input tensor
#     """

#     def __init__(self, n_perturbations: int = 10, *, std: float = 0.1):
#         self.n_perturbations = n_perturbations
#         self.std = std

#     def perturb(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
#         noise = torch.randn(self.n_perturbations, *inputs.shape, device=inputs.device) * self.std
#         return inputs + noise, None
