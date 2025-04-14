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
# TODO : remake all the docstrings of this file to fit with new method signatures

from __future__ import annotations

import itertools
from collections.abc import Iterable, MutableMapping
from functools import singledispatchmethod
from typing import Any

import torch

from interpreto.commons.granularity import GranularityLevel


class Perturbator:
    """
    Base class for perturbators
    If this class is instanciated, it behaves as a no-op perturbator
    Perturbators can be defined by subclassing this class and implementing one (or many) of the following methods :
    - perturb_ids
    - perturb_embeds
    """

    __slots__ = ("inputs_embedder",)

    def __init__(self, inputs_embedder: torch.nn.Module | None = None):
        # Embedders is optional
        self.inputs_embedder = inputs_embedder

    # TODO : this function is replicated in the inference wrapper, enventually merge them
    def _embed(self, model_inputs: MutableMapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
        """
        Embed the inputs using the inputs_embedder

        Args:
            model_inputs (MutableMapping[str, torch.Tensor]): input mapping containing either "input_ids" or "inputs_embeds".

        Raises:
            ValueError: If neither "input_ids" nor "inputs_embeds" are present in the input mapping.

        Returns:
            MutableMapping[str, torch.Tensor]: The input mapping with "inputs_embeds" added.
        """
        # If input embeds are already present, return the unmodified model inputs
        if "inputs_embeds" in model_inputs:
            return model_inputs
        # If no inputs embedder is provided, raise an error
        if self.inputs_embedder is None:
            raise ValueError("Cannot call _embed method from a Perturbator without an inputs embedder")
        # If input ids are present, get the embeddings and add them to the model inputs
        if "input_ids" in model_inputs:
            base_shape = model_inputs["input_ids"].shape
            flatten_embeds = self.inputs_embedder(model_inputs.pop("input_ids").flatten(0, -2))
            model_inputs["inputs_embeds"] = flatten_embeds.view(*base_shape, flatten_embeds.shape[-1])
            return model_inputs
        # If neither input ids nor input embeds are present, raise an error
        raise ValueError("model_inputs should contain either 'input_ids' or 'inputs_embeds'")

    @singledispatchmethod
    def perturb(self, inputs: Any) -> list[tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]]:
        raise NotImplementedError(
            f"Method perturb not implemented for type {type(inputs)} in {self.__class__.__name__}"
        )

    @perturb.register(MutableMapping)
    def _perturb_mapping(
        self, inputs: MutableMapping[str, torch.Tensor]
    ) -> list[tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]]:
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
        mask = None
        if "input_ids" in inputs:
            # Call the tokens perturbation on the inputs ids
            inputs, mask = self.perturb_ids(inputs)

        try:
            # TODO : perform smart combination of perturbation masks on ids and on embeddings !
            # inputs, ids_pert_mask = self.perturb_embeds(self._embed(inputs))
            # final_mask = some_combination(ids_pert_mask, embeds_pert_mask) # something like a elementwise binary or on the tensors ?
            # return inputs, final_mask
            return [self.perturb_embeds(self._embed(inputs))]
        except (ValueError, NotImplementedError):
            return [(inputs, mask)]

    @perturb.register(Iterable)
    def _perturb_iterable(
        self, inputs: Iterable[MutableMapping[str, torch.Tensor]]
    ) -> list[tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]]:
        return list(itertools.chain.from_iterable(self._perturb_mapping(item) for item in inputs))

    def perturb_ids(
        self, model_inputs: MutableMapping
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
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

    def perturb_embeds(
        self, model_inputs: MutableMapping
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        # inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        raise NotImplementedError(f"No way to perturb input embeddings has been defined in {self.__class__.__name__}")


class MaskBasedPerturbator(Perturbator):
    """
    Base class for methods applying a mask to the input
    This class is just furnishing a default implementation for the apply_mask method
    This class should not be subclasses by perturbation methods.
    Please consider using TokenMaskBasedPerturbator or EmbeddingsMaskBasedPerturbator instead, depending on where you want to apply your mask
    """

    __slots__ = ()

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

    __slots__ = ("n_perturbations", "replace_token_id", "granularity_level")

    def __init__(
        self,
        replace_token_id: int,
        inputs_embedder: torch.nn.Module | None = None,
        n_perturbations: int = 1,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
    ):
        super().__init__(inputs_embedder=inputs_embedder)

        # number of perturbations made by the "perturb" method
        self.n_perturbations = n_perturbations

        # token id used to replace the masked tokens
        self.replace_token_id = replace_token_id

        # granularity level of the perturbation (token masking, word masking...)
        # in most commons cases, this should be set to GranularityLevel.TOKEN
        self.granularity_level = granularity_level

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
        perturbation_dimension = int(GranularityLevel.get_length(model_inputs, self.granularity_level).max().item())
        batch_size = model_inputs["input_ids"].shape[0]
        gran_mask = self.get_mask(batch_size, perturbation_dimension)
        model_inputs["mask"] = gran_mask
        gran_assoc_matrix = GranularityLevel.get_association_matrix(model_inputs, self.granularity_level)
        return self.get_real_mask_from_gran_mask(gran_mask, gran_assoc_matrix)

    def perturb_ids(
        self, model_inputs: MutableMapping
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        """
        Method called to perturb the inputs of the model

        Args:
            model_inputs (MutableMapping): mapping given by the tokenizer

        Returns:
            tuple: model_inputs with perturbations and the specific granularity mask
        """
        batch_size = model_inputs["input_ids"].shape[0]
        mask_dim = int(GranularityLevel.get_length(model_inputs, self.granularity_level).max().item())
        gran_mask = self.get_mask(batch_size, mask_dim)
        real_mask = self.get_real_mask_from_gran_mask(
            gran_mask, GranularityLevel.get_association_matrix(model_inputs, self.granularity_level)
        )

        # real_mask, gran_mask = self.get_model_inputs_mask(model_inputs)

        model_inputs["input_ids"] = (
            self.apply_mask(
                inputs=model_inputs["input_ids"].unsqueeze(-1),
                mask=real_mask,
                mask_value=torch.Tensor([self.replace_token_id]),
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

    __slots__ = ("replacement_vector", "n_perturbations")

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        n_perturbations: int = 1,
        replacement_vector: torch.Tensor | None = None,
    ):
        super().__init__(inputs_embedder=inputs_embedder)
        self.n_perturbations = n_perturbations
        self.replacement_vector = replacement_vector

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

    def perturb_embeds(
        self, model_inputs: MutableMapping
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        replacement_vector = self.replacement_vector
        if replacement_vector is None:
            replacement_vector = torch.zeros(
                model_inputs["inputs_embeds"].shape[-1], device=model_inputs["inputs_embeds"].device
            )

        embeddings = model_inputs["inputs_embeds"]
        mask = self.get_mask(embeddings)
        model_inputs["inputs_embeds"] = self.apply_mask(embeddings, mask, replacement_vector)
        return model_inputs, mask


class OcclusionPerturbator(TokenMaskBasedPerturbator):
    """
    Basic class for occlusion perturbations
    """

    __slots__ = ()

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        granularity_level: GranularityLevel = GranularityLevel.TOKEN,
        replace_token_id: int = 0,
    ):
        super().__init__(replace_token_id, inputs_embedder=inputs_embedder, n_perturbations=-1)
        self.granularity_level = granularity_level

    def get_mask(self, num_sequences: int, mask_dim: int) -> torch.Tensor:
        # TODO : use torch diag embed of attention mask instead of torch eye
        return torch.eye(mask_dim).unsqueeze(0).repeat(num_sequences, 1, 1)


class GaussianNoisePerturbator(Perturbator):
    """
    Perturbator adding gaussian noise to the input tensor
    """

    __slots__ = ("n_perturbations", "std")

    def __init__(self, inputs_embedder: torch.nn.Module | None = None, n_perturbations: int = 10, *, std: float = 0.1):
        super().__init__(inputs_embedder)
        self.n_perturbations = n_perturbations
        self.std = std

    def perturb_embeds(
        self, model_inputs: MutableMapping
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor | None]:
        model_inputs["input_embeds"] = model_inputs["input_embeds"].unsqueeze(1).repeat(1, self.n_perturbations, 1, 1)
        model_inputs["attention_mask"] = (
            model_inputs["attention_mask"].unsqueeze(1).repeat(1, self.n_perturbations, 1, 1)
        )
        # add noise
        model_inputs["input_embeds"] += torch.randn_like(model_inputs["input_embeds"]) * self.std
        return model_inputs, None  # return noise ? noise.bool().long() ?
