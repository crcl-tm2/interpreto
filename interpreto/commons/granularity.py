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
Definition of different granularity levels for explainers (tokens, words, sentences...)
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import torch


class GranularityLevel(Enum):
    """
    Enumerations of the different granularity levels supported for masking perturbations
    Allows to define token-wise masking, word-wise masking...
    """

    ALL_TOKENS = "all_tokens"  # All tokens, including special tokens like padding, eos, cls, etc.
    TOKEN = "token"  # Strictly tokens of the input
    WORD = "word"  # Words of the input
    DEFAULT = TOKEN

    @staticmethod
    def __all_tokens_assoc_matrix(tokens_ids: Mapping[str, torch.Tensor]):
        n, l_p = tokens_ids["input_ids"].shape
        return torch.eye(l_p).unsqueeze(0).expand(n, -1, -1)

    @staticmethod
    def __token_assoc_matrix(tokens_ids: Mapping[str, torch.Tensor]):
        # TODO : remake this using only tensor operation (if possible ?)
        n, l_p = tokens_ids["input_ids"].shape
        perturbable_matrix = torch.diag_embed(1 - tokens_ids["special_tokens_mask"])
        non_empty_rows_mask = perturbable_matrix.sum(dim=2) != 0
        l_t = non_empty_rows_mask.sum(dim=1)
        result = torch.zeros(n, l_t.max(), l_p)
        for i in range(n):
            result[i, : l_t[i], :] = perturbable_matrix[i, non_empty_rows_mask[i]]
        return result

    @staticmethod
    def __word_assoc_matrix(tokens_ids: Mapping[str, torch.Tensor]):
        n, l_p = tokens_ids["input_ids"].shape
        l_t = GranularityLevel.get_length(tokens_ids, GranularityLevel.WORD).max()
        index_tensor = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor([a if a is not None else l_t for a in elem])
                for elem in [tokens_ids.word_ids(i) for i in range(n)]
            ],
            batch_first=True,
            padding_value=l_t + 1,
        )
        reference = torch.diagonal_scatter(torch.zeros(l_t + 1, l_t), torch.ones(l_t))
        res = (
            torch.index_select(reference, 0, index_tensor.flatten())
            .reshape(index_tensor.shape + (reference.shape[1],))
            .transpose(-1, -2)
        )
        return res

    @staticmethod
    def get_length(
        tokens_ids: Mapping[str, torch.Tensor], granularity_level: GranularityLevel = DEFAULT
    ) -> torch.Tensor:
        """
        Returns the length of the sequences according to the granularity level

        Args:
            tokens_ids (Mapping[str, torch.Tensor]): tensors to measure
            granularity_level (GranularityLevel, optional): granularity level. Defaults to DEFAULT.

        Returns:
            torch.Tensor: length of the sequences
        """
        match granularity_level:
            case GranularityLevel.ALL_TOKENS:
                return tokens_ids["input_ids"].shape[1] * torch.ones(tokens_ids["input_ids"].shape[0])
            case GranularityLevel.TOKEN:
                return (1 - tokens_ids["special_tokens_mask"]).sum(dim=1)
            case GranularityLevel.WORD:
                return (
                    torch.tensor(
                        [
                            max(filter(lambda x: x is not None, tokens_ids.word_ids(i)))
                            for i in range(tokens_ids["input_ids"].shape[0])
                        ]
                    )
                    + 1
                )
            case _:
                raise NotImplementedError(f"Granularity level {granularity_level} not implemented")

    @staticmethod
    def get_decomposition(tokens_ids: Mapping[str, torch.Tensor], granularity_level: GranularityLevel = DEFAULT):
        match granularity_level:
            case GranularityLevel.ALL_TOKENS:
                return [[[id,] for id in elem] for elem in tokens_ids["input_ids"]]
            case GranularityLevel.TOKEN:
                return [[[id,] for id, mask in zip(elem, spe_tok_mask, strict=True) if mask == 0] for elem, spe_tok_mask in zip(tokens_ids["input_ids"], tokens_ids["special_tokens_mask"], strict=True)]
            case GranularityLevel.WORD:
                # TODO : refaire ça
                res = []
                for index, token_ids in enumerate(tokens_ids["input_ids"]):
                    word_ids = tokens_ids.word_ids(index)
                    res.append([[] for _ in range(max(filter(lambda x: x is not None, word_ids)) + 1)])
                    for tok, word_id in zip(token_ids, word_ids, strict=True):
                        if word_id is not None:
                            res[-1][word_id] += [tok.item()]
                return res
            case _:
                raise NotImplementedError(f"Granularity level {granularity_level} not implemented in decompose function")
        

    @staticmethod
    def get_association_matrix(
        tokens_ids: Mapping[str, torch.Tensor], granularity_level: GranularityLevel = DEFAULT
    ) -> torch.Tensor:
        """
        Creates the matrix to pass from one granularity level to ALL_TOKENS granularity level (finally used by the perturbator)


        Args:
            tokens_ids (Mapping[str, torch.Tensor]): inputs of the perturb meth
            granularity_level (GranularityLevel | None, optional): source granularity level. Defaults to GranularityLevel.DEFAULT.

        Raises:
            NotImplementedError: if granularity level is unknown, raises NotImplementedError

        Returns:
            torch.Tensor: the matrix used to transform a specific granularity mask to a general mask that can be used on tokens
        """
        match granularity_level:
            case GranularityLevel.ALL_TOKENS:
                return GranularityLevel.__all_tokens_assoc_matrix(tokens_ids)
            case GranularityLevel.TOKEN:
                return GranularityLevel.__token_assoc_matrix(tokens_ids)
            case GranularityLevel.WORD:
                return GranularityLevel.__word_assoc_matrix(tokens_ids)
            case _:
                raise NotImplementedError(f"Granularity level {granularity_level} not implemented")
