"""
Simple occlusion perturbations for tokenized inputs
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import singledispatchmethod

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import TokenPerturbation

# TODO : Add a mixin in perturbationts.base to avoid code duplication between multiples token-wise or word-wise perturbations
# TODO : tests pour les méthodes d'occlusion


class TokenOcclusionPerturbator(TokenPerturbation):
    """
    Perturbator removing tokens from the input
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, inputs_embeddings: torch.nn.Module, mask_value: str | None = None
    ):
        # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
        self.tokenizer = tokenizer
        self.inputs_embeddings = inputs_embeddings
        self.mask_value = mask_value or tokenizer.mask_token

    @singledispatchmethod
    def perturb(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"Type {type(inputs)} not supported")

    @perturb.register(str)
    def _(self, inputs: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Get tokens from str input
        tokens = self.tokenizer.tokenize(inputs)
        n_perturbations = len(tokens)
        # Create variations by masking each token
        variations = [
            self.tokenizer.convert_tokens_to_ids(tokens[:i] + [self.mask_value] + tokens[i + 1 :])
            for i in range(n_perturbations)
        ]
        # Get words embeddings for each variation
        embeddings = torch.stack([self.inputs_embeddings(torch.tensor(variation)) for variation in variations])
        # Return embeddings and identity matrix as mask
        return embeddings, torch.eye(n_perturbations)

    @perturb.register(Iterable)
    def _(self, inputs: Iterable):
        # Perturb a batch of inputs (or nested batchs of inputs)
        return [self.perturb(input) for input in inputs]


class WordOcclusionPerturbator(TokenPerturbation):
    """
    Perturbator removing words from the input
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, inputs_embeddings: torch.nn.Module, mask_value: str | None = None
    ):
        # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
        self.tokenizer = tokenizer
        self.inputs_embeddings = inputs_embeddings
        self.mask_value = mask_value or tokenizer.mask_token

    @singledispatchmethod
    def perturb(self, inputs) -> tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError(f"Type {type(inputs)} not supported")

    @perturb.register(str)
    def _(self, inputs: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Get tokens from str input
        words = inputs.split()
        n_perturbations = len(words)
        # Create variations by masking each word
        variations = []
        for word in inputs.split():
            sentence = "".join(inputs.replace(word, self.mask_value))
            tokens = self.tokenizer.tokenize(sentence)
            variations.append(self.tokenizer.convert_tokens_to_ids(tokens))
        # Get words embeddings for each variation
        embeddings = torch.stack([self.inputs_embeddings(torch.tensor(variation)) for variation in variations])
        # Return embeddings and identity matrix as mask
        return embeddings, torch.eye(n_perturbations)

    @perturb.register(Iterable)
    def _(self, inputs: Iterable) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # TODO : put this in a mixin to avoid code duplication
        # Perturb a batch of inputs (or nested batchs of inputs)
        return [self.perturb(input) for input in inputs]
