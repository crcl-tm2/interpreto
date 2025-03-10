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

from collections.abc import Iterable, Mapping
from enum import Enum
from functools import singledispatchmethod

import torch
from transformers import PreTrainedTokenizer


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
    def __all_tokens_assoc_matrix(tokens_ids):
        n, l_p = tokens_ids["input_ids"].shape
        return torch.eye(l_p).unsqueeze(0).expand(n, -1, -1)

    @staticmethod
    def __token_assoc_matrix(tokens_ids):
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
    def __word_assoc_matrix(tokens_ids):
        # TODO : implement word granularity level
        raise NotImplementedError("Word granularity level not implemented")

    @staticmethod
    def get_length(tokens_ids:Mapping[str, torch.Tensor], granularity_level: GranularityLevel = DEFAULT) -> torch.Tensor:
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
                return tokens_ids["input_ids"].shape[1]
            case GranularityLevel.TOKEN:
                return (1 - tokens_ids["special_tokens_mask"]).sum(dim=1)
            case GranularityLevel.WORD:
                # TODO : implement word granularity level
                raise NotImplementedError("Word granularity level not implemented")
            case _:
                raise NotImplementedError(f"Granularity level {granularity_level} not implemented")

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

class BasePerturbator:
    """
    Base class for perturbators
    If this class is instanciated, it behaves as a no-op perturbator
    Perturbators can be defined by subclassing this class and implementing one (or many) of the following methods :
    - perturb_strings
    - perturb_ids
    - perturb_tensors
    """

    def __init__(self, tokenizer: PreTrainedTokenizer | None = None, inputs_embedder: torch.nn.Module | None = None):
        # Tokenizer and embedders are optional
        self.tokenizer = tokenizer
        self.inputs_embedder = inputs_embedder

    @singledispatchmethod
    def perturb(self, inputs) -> Mapping[str, torch.Tensor]:
        """
        Main method called to perturb an input before giving it to a model
        Output is a mapping that can be passed directly to the model as kwargs

        Args:
            inputs (_type_): inputs given to the perturbator. It can be one of the following types :
             - str|Iterable[str] : a single string or a collection of strings to perturb
             - Mapping[str, torch.Tensor] : a mapping of tensors to perturb, with keys similar to the keys found in the output of a tokenizer.
                                                            Direct output of a tokenizer can be passed to this method
             - torch.Tensor : a tensor to perturb. This is the case when the perturbator is used to perturb embeddings

        Raises:
            NotImplementedError: If the perturbator receives a type different from the ones listed above

        Returns:
            Mapping[str, torch.Tensor]: A mapping of tensors that can be passed directly to the model as kwargs
        """
        raise NotImplementedError(
            f"Method perturb not implemented for type {type(inputs)} in {self.__class__.__name__}"
        )

    @perturb.register(str)
    def _(self, inputs: str) -> Mapping[str, torch.Tensor]:
        """
        perturbation of a single string (transformed as a collection of 1 string for convenience)

        Args:
            inputs (str): string to perturb
        """
        return self.perturb([inputs])

    @perturb.register(Iterable)
    def _(self, inputs: Iterable[str]) -> Mapping[str, torch.Tensor]:
        """
        perturbation of a single string (transformed as a collection of 1 string for convenience)

        Args:
            inputs (Iterable[str]): collection of strings to perturb
        """
        # If no tokenizer is provided, we raise an error
        if self.tokenizer is None:
            raise ValueError(
                "A tokenizer is required to perturb strings. Please provide a tokenizer when initializing the perturbator or specify it with 'perturbator.tokenizer = some_tokenizer'"
            )
        # call string perturbation method on given strings
        perturbed_strings = self.perturb_strings(inputs)

        # Call the tokenizer on the produced strings
        tokens = self.tokenizer(
            perturbed_strings,
            truncation=True,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        # Call the next perturbation step (identity if no further perturbation has been defined)
        return self.perturb(tokens)

    @perturb.register(Mapping)
    def _(self, inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        Method called when we ask the perturbator to perturb a mapping of tensors, generally the output of a tokenizer
        The mapping should be similar to mappings returned by the tokenizer.
        It should at least have "input_ids", "attention_mask" and "offset_mappings" keys
        Give directly the output of the tokenizer without modifying it would be the best and most common way to use this method

        Args:
            inputs (Mapping[str, torch.Tensor]): output of the tokenizers
        """
        assert "offset_mapping" in inputs, (
            "Offset mapping is required to perturb tokens, specify the 'return_offsets_mapping=True' parameter when tokenizing the input"
        )

        # Call the tokens perturbation on the inputs ids
        inputs = self.perturb_ids(inputs)

        # If no inputs have been provided, return the perturbed ids
        if self.inputs_embedder is None:
            return inputs
        # Check if an embedding perturbation has been defined
        try:
            # If perturb_tensors has been defined, call it on the embeddings
            embeddings = self.perturb_tensors(self.inputs_embedder(inputs))
            return {"inputs_embeds": embeddings}  # add complementary data in dict
        except NotImplementedError:
            # If no embeddings perturbation has been defined to the, return the perturbed ids
            return inputs

    @perturb.register(torch.Tensor)
    def _(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """
        Method called when we ask the perturbator to perturb a tensor, generally embeddings

        Args:
            inputs (torch.Tensor): inputs embeddings to perturb
        """
        return {"inputs_embeds": self.perturb_tensors(inputs)}

    # Methods to be implemented by subclasses:
    def perturb_strings(self, strings: Iterable[str]) -> Iterable[str]:
        """
        Perturb a sequence of strings

        Args:
            strings (Iterable[str]): sequence of strings to perturb

        Returns:
            Iterable[str]: New sequence of strings after perturbation
        """
        return strings

    def perturb_ids(self, model_inputs: Mapping) -> Mapping[str, torch.Tensor]:
        """
        Perturb the input of the model

        Args:
            model_inputs (Mapping): Mapping given by the tokenizer

        Returns:
            Mapping[str, torch.Tensor]: Perturbed mapping
        """
        return model_inputs

    def perturb_tensors(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Perturb the embeddings

        Args:
            tensors (torch.Tensor): inputs embeddings that may be given to the model
            shape should be (batch_size, sequence_length, embedding_size)

        Raises:
            NotImplementedError: If this method is called without being implemented in the subclass, it will raise an error

        Returns:
            torch.Tensor: perturbed embeddings, shape would be (batch_size, n_perturbations, sequence_length, embedding_size)
        """
        raise NotImplementedError(f"No way to perturb embeddings has been defined in {self.__class__.__name__}")


class MaskBasedPerturbator(BasePerturbator):
    """
    Base class for methods applying a mask to the input
    This class is just furnishing a default implementation for the apply_mask method
    This class should not be subclasses by perturbation methods.
    Please consider using TokenMaskBasedPerturbator or EmbeddingsMaskBasedPerturbator instead, depending on where you want to apply your mask
    """

    def apply_mask(self, inputs: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor) -> torch.Tensor:
        """
        Basic mask application method

        Args:
            inputs (torch.Tensor): inputs to mask
            mask (torch.Tensor): mask matrix to apply
            mask_value (torch.Tensor): tensor used as a mask (mask token, zero tensor, etc.)

        Returns:
            torch.Tensor: masked inputs
        """
        # TODO generalize to upper dimensions for other types of input data
        base = torch.einsum("nld,npl->npld", inputs, 1 - mask)
        masked = torch.einsum("npl,d->npld", mask, mask_value)
        return base + masked


class TokenMaskBasedPerturbator(MaskBasedPerturbator):
    """
    Base class for perturbations consisting in applying masks on token (or groups of tokens)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        n_perturbations: int = 1,
        mask_token: str = None,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
    ):
        super().__init__(tokenizer=tokenizer, inputs_embedder=inputs_embedder)

        # number of perturbations made by the "perturb" method
        self.n_perturbations = n_perturbations
        # TODO : set general mask token if no tokenizer is given
        self.mask_token = mask_token or self.tokenizer.mask_token if self.tokenizer is not None else "[MASK]"

        # granularity level of the perturbation (token masking, word masking...)
        # in most commons cases, this should be set to GranularityLevel.TOKEN
        self.granularity_level = granularity_level

    @property
    def mask_token_id(self) -> int:
        """
        Returns the mask token id
        """
        return self.tokenizer.convert_tokens_to_ids(self.mask_token)

    # TODO : rename "perturbation_dimension" to better name
    def get_mask(self, batch_size, perturbation_dimension)->torch.Tensor:#self, model_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Method returning a perturbation mask for a given set of inputs
        This method should be implemented in subclasses

        The created mask should be of size (batch_size, n_perturbations, mask_dimension)
        where mask_dimension is the length of the sequence according to the granularity level (number of tokens, number of words, number of sequences...)

        Args:
            model_inputs (Mapping[str, torch.Tensor]): mapping given by the tokenizer

        Returns:
            torch.Tensor: mask to apply
        """
        # Exemple implementation that returns a no-perturbation mask
        return torch.zeros((batch_size, self.n_perturbations, perturbation_dimension))

    @staticmethod
    def get_gran_mask_from_real_mask(
        model_inputs: Mapping[str, torch.Tensor],
        real_mask: torch.Tensor,
        granularity_level: GranularityLevel = GranularityLevel.DEFAULT,
    ) -> torch.Tensor:
        """
        Transforms a real token-wise mask to an approximation of it's associated mask for a certain granularity level

        Args:
            model_inputs (Mapping[str, torch.Tensor]): mapping given by the tokenizer
            p_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: granularity level mask
        """
        # TODO : eventually store gran matrix in tokens to avoid recomputing it ?
        t_gran_matrix = GranularityLevel.get_association_matrix(model_inputs, granularity_level).transpose(-1, -2)
        return torch.einsum("npr,ntr->npt", real_mask, t_gran_matrix) / t_gran_matrix.sum(dim=-1)

    @staticmethod
    def get_real_mask_from_gran_mask(
        model_inputs: Mapping[str, torch.Tensor], gran_mask: torch.Tensor, granularity_level: GranularityLevel.DEFAULT
    ) -> torch.Tensor:
        """
        Transforms a specific granularity mask to a general token-wise mask

        Args:
            model_inputs (Mapping[str, torch.Tensor]): mapping given by the tokenizer
            t_mask (torch.Tensor): mask defined at a certain granularity level

        Returns:
            torch.Tensor: real general mask
        """
        gran_matrix = GranularityLevel.get_association_matrix(model_inputs, granularity_level)
        # TODO : eventually store gran matrix in tokens to avoid recomputing it ?
        return torch.einsum("npt,ntr->npr", gran_mask, gran_matrix)

    def get_model_inputs_mask(self, model_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Method returning the real mask to apply on the model inputs
        This method may be overriden in subclasses to provide a more specific mask
        default implementation gets the real mask from the specific granularity mask

        Args:
            model_inputs (Mapping[str, torch.Tensor]): mapping given by the tokenizer

        Returns:
            torch.Tensor: real general mask
        """
        perturbation_dimension = GranularityLevel.get_length(model_inputs, self.granularity_level).max()
        batch_size = model_inputs["input_ids"].shape[0]
        gran_mask = self.get_mask(batch_size, perturbation_dimension)
        model_inputs["mask"] = gran_mask
        return self.get_real_mask_from_gran_mask(model_inputs, gran_mask, self.granularity_level)

    def perturb_ids(self, model_inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        Method called to perturb the inputs of the model

        Args:
            model_inputs (Mapping[str, torch.Tensor]): _description_

        Returns:
            dict[str, torch.Tensor]: _description_
        """
        real_mask = self.get_model_inputs_mask(model_inputs)

        model_inputs["input_ids"] = self.apply_mask(
            inputs=model_inputs["input_ids"].unsqueeze(-1),
            mask=real_mask,
            mask_value=torch.Tensor([self.mask_token_id]),
        ).squeeze(-1)
        return model_inputs


class EmbeddingsMaskBasedPerturbator(MaskBasedPerturbator):
    """
    Base class for perturbations consisting in applying masks on embeddings
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        n_perturbations: int = 1,
        default_mask_vector: torch.Tensor = None,
    ):
        super().__init__(tokenizer=tokenizer, inputs_embedder=inputs_embedder)
        self.n_perturbations = n_perturbations
        self.default_mask_vector = default_mask_vector or torch.zeros(self.inputs_embedder.embedding_dim)

    def get_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Method returning a perturbation mask for a given set of embeddings
        This method should be implemented in subclasses

        Args:
            embeddings (torch.Tensor): embeddings to perturb

        Returns:
            torch.Tensor: mask to apply
        """
        raise NotImplementedError(f"Method get_mask not implemented in {self.__class__.__name__}")

    def perturb_embeddings(self, embeddings):
        mask = self.get_mask(embeddings)
        perturbed_embeddings = self.apply_mask(embeddings, mask, self.default_mask_vector)
        return {"real_mask": mask, "inputs_embeds": perturbed_embeddings}


class OcclusionPerturbator(TokenMaskBasedPerturbator):
    """
    Basic class for occlusion perturbations
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
    ):
        super().__init__(tokenizer=tokenizer, inputs_embedder=inputs_embedder, n_perturbations=-1)
        self.granularity_level = granularity_level

    def get_mask(self, batch_size, perturbation_dimension)->torch.Tensor:
        # TODO : use torch diag embed of attention mask instead of torch eye
        return torch.eye(perturbation_dimension).unsqueeze(0).repeat(batch_size, 1, 1)

        #assoc_matrix = GranularityLevel.get_association_matrix(model_inputs, self.granularity_level)
        #return torch.diag_embed(torch.einsum("ntl,nl->nt", assoc_matrix, model_inputs["attention_mask"].float()))


class GaussianNoisePerturbator(BasePerturbator):
    """
    Perturbator adding gaussian noise to the input tensor
    """

    def __init__(self, n_perturbations: int = 10, *, std: float = 0.1):
        super().__init__(None, None)
        self.n_perturbations = n_perturbations
        self.std = std

    def perturb_tensors(self, tensors: torch.Tensor) -> torch.Tensor:
        perturbations = tensors.unsqueeze(1).repeat(1, self.n_perturbations, 1, 1)
        noise = torch.randn_like(perturbations) * self.std
        return perturbations + noise
