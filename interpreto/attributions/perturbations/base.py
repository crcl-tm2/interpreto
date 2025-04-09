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

from collections.abc import Iterable, MutableMapping
from functools import singledispatchmethod

import torch
from transformers import PreTrainedTokenizer

from interpreto.commons.granularity import GranularityLevel


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
    def perturb(self, inputs) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        """
        Main method called to perturb an input before giving it to a model
        Output is a mapping that can be passed directly to the model as kwargs

        Args:
            inputs (_type_): inputs given to the perturbator. It can be one of the following types :
             - str|Iterable[str] : a single string or a collection of strings to perturb
             - MutableMapping[str, torch.Tensor] : a mapping of tensors to perturb, with keys similar to the keys found in the output of a tokenizer.
                                                            Direct output of a tokenizer can be passed to this method
             - torch.Tensor : a tensor to perturb. This is the case when the perturbator is used to perturb embeddings

        Raises:
            NotImplementedError: If the perturbator receives a type different from the ones listed above

        Returns:
            MutableMapping[str, torch.Tensor]: A mapping of tensors that can be passed directly to the model as kwargs
        """
        raise NotImplementedError(
            f"Method perturb not implemented for type {type(inputs)} in {self.__class__.__name__}"
        )

    @perturb.register(str)
    def _(self, inputs: str) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        """
        perturbation of a single string (transformed as a collection of 1 string for convenience)

        Args:
            inputs (str): string to perturb
        """
        return self.perturb([inputs])

    @perturb.register(Iterable)
    def _(self, inputs: Iterable[str]) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
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

        # Call the tokenizer on the produced strings
        tokens = self.tokenizer(
            inputs,
            truncation=True,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        # Call the next perturbation step (identity if no further perturbation has been defined)
        return self.perturb(tokens)

    @perturb.register(MutableMapping)
    def _(self, inputs: MutableMapping[str, torch.Tensor]) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        """
        Method called when we ask the perturbator to perturb a mapping of tensors, generally the output of a tokenizer
        The mapping should be similar to mappings returned by the tokenizer.
        It should at least have "input_ids", "attention_mask" and "offset_mappings" keys
        Give directly the output of the tokenizer without modifying it would be the best and most common way to use this method

        Args:
            inputs (MutableMapping[str, torch.Tensor]): output of the tokenizers
        """
        assert "offset_mapping" in inputs, (
            "Offset mapping is required to perturb tokens, specify the 'return_offsets_mapping=True' parameter when tokenizing the input"
        )

        # Call the tokens perturbation on the inputs ids
        inputs, mask = self.perturb_ids(inputs)

        # If no inputs have been provided, return the perturbed ids
        if self.inputs_embedder is None:
            return inputs, mask
        # Check if an embedding perturbation has been defined
        try:
            # If perturb_tensors has been defined, call it on the embeddings

            embeddings, perturbation_mask = self.perturb_tensors(self.inputs_embedder(inputs))
            # TODO : perform smart combination of perturbation masks
            attention_mask = inputs["attention_mask"].unsqueeze(1).repeat(1, embeddings.shape[1], 1)

            return {
                "inputs_embeds": embeddings,
                "attention_mask": attention_mask,
            }, perturbation_mask  # add complementary data in dict
        except NotImplementedError:
            # If no embeddings perturbation has been defined to the, return the perturbed ids
            return inputs, mask

    @perturb.register(torch.Tensor)
    def _(self, inputs: torch.Tensor) -> MutableMapping[str, torch.Tensor]:
        """
        Method called when we ask the perturbator to perturb a tensor, generally embeddings

        Args:
            inputs (torch.Tensor): inputs embeddings to perturb
        """
        perturbed_tensor, mask = self.perturb_tensors(inputs)
        return {"inputs_embeds": perturbed_tensor, "attention_mask": torch.ones(perturbed_tensor.shape[:-1])}, mask

    def perturb_ids(self, model_inputs: MutableMapping) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        """
        Perturb the input of the model

        Args:
            model_inputs (MutableMapping): Mapping given by the tokenizer

        Returns:
            MutableMapping[str, torch.Tensor]: Perturbed mapping
        """
        # add perturbation dimension
        model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(1)
        return model_inputs, torch.zeros_like(model_inputs["input_ids"])

    def perturb_tensors(self, tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        Basic mask application method.

        If last dimension `d` is 1 (in case of tokens and not embeddings), this last dimension will be squeezed out
        and the returned tensor will have shape (num_sequences, n_perturbations, mask_dim).

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
        return (base + masked).squeeze(-1)


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

    def get_mask(self, num_sequences: int, mask_dim: int) -> torch.Tensor:
        """
        Method returning a perturbation mask for a given set of inputs
        This method should be implemented in subclasses

        The created mask should be of size (batch_size, n_perturbations, mask_dim)
        where mask_dim is the length of the sequence according to the granularity level (number of tokens, number of words, number of sentences...)

        Args:
            num_sequences (int): number of sequences
            mask_dim (int): length of the sequence according to the granularity level

        Returns:
            torch.Tensor: mask to apply on the inputs, of shape (num_sequences, n_perturbations, mask_dim)
        """
        # Exemple implementation that returns a no-perturbation mask
        return torch.zeros(num_sequences, self.n_perturbations, mask_dim)

    @staticmethod
    def get_gran_mask_from_real_mask(
        model_inputs: MutableMapping[str, torch.Tensor],
        real_mask: torch.Tensor,
        granularity_level: GranularityLevel = GranularityLevel.DEFAULT,
    ) -> torch.Tensor:
        """
        Transforms a real token-wise mask to an approximation of its associated mask for a certain granularity level

        Args:
            model_inputs (MutableMapping[str, torch.Tensor]): mapping given by the tokenizer
            p_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: granularity level mask
        """
        # TODO : eventually store gran matrix in tokens to avoid recomputing it ?
        t_gran_matrix = GranularityLevel.get_association_matrix(model_inputs, granularity_level).transpose(-1, -2)
        return torch.einsum("npr,ntr->npt", real_mask, t_gran_matrix) / t_gran_matrix.sum(dim=-1)

    @staticmethod
    def get_real_mask_from_gran_mask(gran_mask: torch.Tensor, gran_assoc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Transforms a specific granularity mask to a general token-wise mask

        Args:
            model_inputs (MutableMapping[str, torch.Tensor]): mapping given by the tokenizer
            gran_assoc_matrix (torch.Tensor): association matrix for a specific granularity level

        Returns:
            torch.Tensor: real general mask
        """
        # TODO : eventually store gran matrix in tokens to avoid recomputing it ?
        return torch.einsum("npt,ntr->npr", gran_mask, gran_assoc_matrix)

    def get_model_inputs_mask(self, model_inputs: MutableMapping) -> torch.Tensor:
        """
        Method returning the real mask to apply on the model inputs
        This method may be overriden in subclasses to provide a more specific mask
        default implementation gets the real mask from the specific granularity mask

        Args:
            model_inputs (MutableMapping): mapping given by the tokenizer

        Returns:
            torch.Tensor, torch.Tensor: real general mask and specific granularity mask (theoretical mask)
        """
        perturbation_dimension = GranularityLevel.get_length(model_inputs, self.granularity_level).max().int().item()
        batch_size = model_inputs["input_ids"].shape[0]
        gran_mask = self.get_mask(batch_size, perturbation_dimension)
        model_inputs["mask"] = gran_mask
        gran_assoc_matrix = GranularityLevel.get_association_matrix(model_inputs, self.granularity_level)
        return self.get_real_mask_from_gran_mask(gran_mask, gran_assoc_matrix)

    def perturb_ids(self, model_inputs: MutableMapping) -> MutableMapping[str, torch.Tensor]:
        """
        Method called to perturb the inputs of the model

        Args:
            model_inputs (MutableMapping): mapping given by the tokenizer

        Returns:
            tuple: model_inputs with perturbations and the specific granularity mask
        """
        batch_size = model_inputs["input_ids"].shape[0]
        mask_dim = GranularityLevel.get_length(model_inputs, self.granularity_level).max().int().item()
        gran_mask = self.get_mask(batch_size, mask_dim)
        real_mask = self.get_real_mask_from_gran_mask(
            gran_mask, GranularityLevel.get_association_matrix(model_inputs, self.granularity_level)
        )

        # real_mask, gran_mask = self.get_model_inputs_mask(model_inputs)

        model_inputs["input_ids"] = (
            self.apply_mask(
                inputs=model_inputs["input_ids"].unsqueeze(-1),
                mask=real_mask,
                mask_value=torch.Tensor([self.mask_token_id]),
            )
            .squeeze(-1)
            .to(torch.int)
        )

        # Repeat other keys in encoding for each perturbation
        for k in model_inputs.keys():
            if k != "input_ids":
                repeats = [1] * (model_inputs[k].dim() + 1)
                repeats[1] = model_inputs["input_ids"].shape[1]
                model_inputs[k] = model_inputs[k].unsqueeze(1).repeat(*repeats)

        return model_inputs, gran_mask


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

    def get_mask(self, num_sequences: int, mask_dim: int) -> torch.Tensor:
        # TODO : use torch diag embed of attention mask instead of torch eye
        return torch.eye(mask_dim).unsqueeze(0).repeat(num_sequences, 1, 1)


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
