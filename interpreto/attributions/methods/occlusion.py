"""
Simple occlusion perturbations for tokenized inputs
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable
from functools import singledispatchmethod
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.base import InferenceExplainer
from interpreto.attributions.perturbations.base import TokenPerturbator
from interpreto.typing import ModelInput

# TODO : Add a mixin in perturbationts.base to avoid code duplication between multiples token-wise or word-wise perturbations
# TODO : tests pour les méthodes de word occlusion


class _OcclusionPerturbator(TokenPerturbator):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, inputs_embeddings: torch.nn.Module, mask_value: str | None = None
    ):
        # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
        self.tokenizer = tokenizer
        self.inputs_embeddings = inputs_embeddings
        self.mask_value = mask_value or tokenizer.mask_token

    @abstractmethod
    def _perturb_single_element(self, inputs: str) -> tuple[torch.Tensor]:
        """
        Perturb a single sentence by applying token occlusion

        Args:
            inputs (str): sentence to perturb

        Returns:
            tuple[torch.Tensor]: embeddings of perturbed sentence and associated mask
        """
        raise NotImplementedError

    @singledispatchmethod
    def perturb(self, inputs) -> tuple[torch.Tensor] | tuple[list[torch.Tensor]]:
        """
        Perturb a sentence or a collection of sentences by applying token occlusion

        Args:
            inputs (str|Iterable[str]): sentence to perturb

        Returns:
            tuple[torch.Tensor]|tuple[list[torch.Tensor]]: embeddings of perturbed sentences and associated masks
        """
        raise NotImplementedError(f"Type {type(inputs)} not supported")

    @perturb.register(str)
    def _(self, inputs: str) -> list[tuple[torch.Tensor]]:
        return [self._perturb_single_element(inputs)]

    @perturb.register(Iterable)
    def _(self, inputs: Iterable) -> list[tuple[torch.Tensor]]:
        # Perturb a batch of inputs (or nested batchs of inputs)
        return [self._perturb_single_element(item) for item in inputs]


class _TokenOcclusionPerturbator(_OcclusionPerturbator):
    """
    Perturbator removing tokens from the input
    """
    def _perturb_single_element(self, inputs: str) -> tuple[torch.Tensor]:
        # Get tokens from str input
        tokens = self.tokenizer.tokenize(inputs)
        n_perturbations = len(tokens)
        # Create variations by masking each token
        variations = [
            self.tokenizer.convert_tokens_to_ids(tokens[:i] + [self.mask_value] + tokens[i + 1 :])
            for i in range(n_perturbations)
        ]
        # Get words embeddings for each variation
        embeddings = torch.stack(
            [self.inputs_embeddings(torch.tensor(variation)) for variation in variations]
        )
        # Return embeddings and identity matrix as mask
        return embeddings.unsqueeze(0), torch.eye(n_perturbations).unsqueeze(0)

class _WordOcclusionPerturbator(_OcclusionPerturbator):
    """
    Perturbator removing words from the input
    """
    def _perturb_single_element(self, inputs: str) -> tuple[torch.Tensor]:
        # Get tokens from str input
        words = inputs.split()
        n_perturbations = len(words)
        # Create variations by masking each word
        variations = []
        for index, word in enumerate(words):
            first_part = self.tokenizer.tokenize(" ".join(words[:index]))
            second_part = self.tokenizer.tokenize(" " + " ".join(words[index + 1 :]))

            tokens = first_part + [self.tokenizer.mask_token for _ in self.tokenizer.tokenize(word)] + second_part
            # add truncation ?
            # tokens = tokens[: max_nb_tokens]
            variations.append(self.tokenizer.convert_tokens_to_ids(tokens))

        max_length = max([len(tokens) for tokens in variations])
        # TODO : put the padding in a separate reusable utility function
        variations = [v + [self.tokenizer.pad_token_id] * (max_length - len(v)) for v in variations]

        # Get words embeddings for each variation
        embeddings = [self.inputs_embeddings(torch.tensor(variation)) for variation in variations]

        # Return embeddings and identity matrix as mask
        return torch.stack(embeddings).unsqueeze(0), torch.eye(n_perturbations).unsqueeze(0)

class OcclusionExplainer(InferenceExplainer):
    def __init__(
        self,
        tokenizer:Callable,
        inference_wrapper: Callable,
        batch_size: int,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            inference_wrapper=inference_wrapper,
            batch_size=batch_size,
            perturbator=_TokenOcclusionPerturbator(
                tokenizer=tokenizer,
                inputs_embeddings=inference_wrapper.model.get_input_embeddings(),
            ),
            aggregator=aggregator,
            device=device,
        )

    def explain(self, inputs: ModelInput, targets) -> Any:
        """
        main process of attribution method
        """
        res = []
        perturbations = self.perturbator.perturb(inputs)
        for (embeddings, mask), target in zip(perturbations, targets):
            # repeat target tensor to create p axis
            target = target.unsqueeze(1).repeat(1, embeddings.shape[1], 1)

            self.inference_wrapper.to(self.device)
            results = self.inference_wrapper.batch_inference(embeddings, target, flatten=True)
            self.inference_wrapper.cpu()  # TODO: check if we need to do this

            explanation = self.aggregator(results, mask)
            res.append(explanation)
        return res
