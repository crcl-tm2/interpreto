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

from interpreto.attributions.aggregations.base import Aggregator, MaskwiseMeanAggregation
from interpreto.attributions.base import InferenceExplainer
from interpreto.attributions.perturbations.base import TokenPerturbator
from interpreto.typing import ModelInput

# TODO : Add a mixin in perturbationts.base to avoid code duplication between multiples token-wise or word-wise perturbations
# TODO : tests pour les méthodes de word occlusion



class TokenOcclusionPerturbator(TokenPerturbator):
    """
    Perturbator removing tokens from the input
    """
    def __init__(
        self, tokenizer: PreTrainedTokenizer,
        inputs_embeddings: torch.nn.Module,
        mask_value: str | None = None
    ):
        # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
        self.tokenizer = tokenizer
        self.inputs_embeddings = inputs_embeddings
        self.mask_value = mask_value or tokenizer.mask_token

    @property
    def mask_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.mask_value)

    @property
    def mask_vector(self):
        return self.inputs_embeddings.weight[self.mask_id]

    def apply_mask(self, inputs:torch.Tensor, mask:torch.Tensor, mask_value:torch.Tensor|None=None):
        mask_value = mask_value or self.mask_vector
        # TODO : eventually replace l and d dims with elipsis to deal with any number of dimensions ?
        base = torch.einsum("nld,npl->npld", inputs, 1 - mask)
        masked = torch.einsum("npl,d->npld", mask, mask_value)
        return base + masked

    def perturb(self, inputs: str|Iterable[str]) -> tuple[torch.Tensor]:
        encoding = self.tokenizer(inputs,
                        truncation=True,
                        return_tensors='pt',
                        padding="max_length",
                        max_length=512,
                        return_offsets_mapping=True)
        tokens_ids = encoding["input_ids"]
        attention_masks = encoding["attention_mask"]
        max_attention_mask_length = attention_masks.sum(dim=1).max().item()
        perturbation_mask = torch.diag_embed(attention_masks)[:, :max_attention_mask_length]
        embeddings = self.inputs_embeddings(tokens_ids)
        result = self.apply_mask(embeddings, perturbation_mask)
        return result, perturbation_mask

# class _OcclusionPerturbator(TokenPerturbator):
#     def __init__(
#         self, tokenizer: PreTrainedTokenizer, inputs_embeddings: torch.nn.Module, mask_value: str | None = None
#     ):
#         # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
#         self.tokenizer = tokenizer
#         self.inputs_embeddings = inputs_embeddings
#         self.mask_value = mask_value or tokenizer.mask_token

#     @abstractmethod
#     def _perturb_single_element(self, inputs: str) -> tuple[torch.Tensor]:
#         """
#         Perturb a single sentence by applying token occlusion

#         Args:
#             inputs (str): sentence to perturb

#         Returns:
#             tuple[torch.Tensor]: embeddings of perturbed sentence and associated mask
#         """
#         raise NotImplementedError

#     @singledispatchmethod
#     def perturb(self, inputs) -> tuple[torch.Tensor] | tuple[list[torch.Tensor]]:
#         """
#         Perturb a sentence or a collection of sentences by applying token occlusion

#         Args:
#             inputs (str|Iterable[str]): sentence to perturb

#         Returns:
#             tuple[torch.Tensor]|tuple[list[torch.Tensor]]: embeddings of perturbed sentences and associated masks
#         """
#         raise NotImplementedError(f"Type {type(inputs)} not supported")

#     @perturb.register(str)
#     def _(self, inputs: str) -> list[tuple[torch.Tensor]]:
#         return [self._perturb_single_element(inputs)]

#     @perturb.register(Iterable)
#     def _(self, inputs: Iterable) -> list[tuple[torch.Tensor]]:
#         # Perturb a batch of inputs (or nested batchs of inputs)
#         return [self._perturb_single_element(item) for item in inputs]


# class _TokenOcclusionPerturbator(_OcclusionPerturbator):
#     """
#     Perturbator removing tokens from the input
#     """
#     def get_mask(self, n:int, p:int, l:int) -> torch.Tensor:
#         return torch.eye(p).unsqueeze(0).repeat(n, 1, 1)

#     def _get_single_mask(self, p, l)->torch.Tensor:
#         assert p <= l, "Number of tokens to mask should be less than the length of the input"
#         result = torch.zeros(p, l)
#         result[:p, :p] = torch.eye(p)
#         return result

#     def apply_mask(self, inputs:torch.Tensor, mask:torch.Tensor, mask_value:torch.Tensor|None=None):
#         mask_value = mask_value or self.inputs_embeddings(self.tokenizer.mask_token_id)
#         # TODO : eventually replace l and d dims with elipsis to deal with any number of dimensions ?
#         base = torch.einsum("nld,npl->npld", inputs, 1 - mask)
#         masked = torch.einsum("npl,d->npld", mask, mask_value)
#         return base + masked

#     def _perturb_single_element(self, inputs: str) -> tuple[torch.Tensor]:
#         encoding = tokenizer(sentences,
#                         truncation=True,
#                         return_tensors='pt',
#                         padding="max_length",
#                         max_length=512,
#                         return_offsets_mapping=True)
        
#         token_ids = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length)["input_ids"]
#         inputs_vector = self.inputs_embeddings(token_ids)
#         n, l, p = inputs_vector.shape
#         mask = self._get_single_mask(p, l)
#         return self.apply_mask(inputs_vector, mask), mask

#         # get id of mask token
#         mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_value)

#         # Get tokens from str input
#         inputs_ids = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=self.tokenizer.model_max_length)["input_ids"]

#         # Generate variations of the sentence by masking each token (except first and last)
#         variations = [inputs_ids[:i] + [self.mask_value] + inputs_ids[i + 1 :] for i in range(1, len(inputs_ids) - 1)]

#         # Get words embeddings for each variation
#         embeddings = torch.stack([self.inputs_embeddings(torch.tensor(variation)) for variation in variations])

#         # Return embeddings and identity matrix as mask
#         mask = torch.zeros(len(variations), len(inputs_ids))
#         mask[:, 1:-1] = 1
#         print(mask)
#         exit()
#         return embeddings.unsqueeze(0), torch.eye(n_perturbations).unsqueeze(0)


# class _WordOcclusionPerturbator(_OcclusionPerturbator):
#     """
#     Perturbator removing words from the input
#     """

#     def _perturb_single_element(self, inputs: str) -> tuple[torch.Tensor]:
#         # Get tokens from str input
#         words = inputs.split()
#         n_perturbations = len(words)
#         # Create variations by masking each word
#         variations = []
#         for index, word in enumerate(words):
#             first_part = self.tokenizer.tokenize(" ".join(words[:index]))
#             second_part = self.tokenizer.tokenize(" " + " ".join(words[index + 1 :]))

#             tokens = first_part + [self.tokenizer.mask_token for _ in self.tokenizer.tokenize(word)] + second_part
#             # add truncation ?
#             # tokens = tokens[: max_nb_tokens]
#             variations.append(self.tokenizer.convert_tokens_to_ids(tokens))

#         max_length = max([len(tokens) for tokens in variations])
#         # TODO : put the padding in a separate reusable utility function
#         variations = [v + [self.tokenizer.pad_token_id] * (max_length - len(v)) for v in variations]

#         # Get words embeddings for each variation
#         embeddings = [self.inputs_embeddings(torch.tensor(variation)) for variation in variations]

#         # Return embeddings and identity matrix as mask
#         return torch.stack(embeddings).unsqueeze(0), torch.eye(n_perturbations).unsqueeze(0)


# class OcclusionExplainer(InferenceExplainer):
#     def __init__(
#         self,
#         tokenizer: Callable,
#         inference_wrapper: Callable,
#         batch_size: int,
#         aggregator: Aggregator=MaskwiseMeanAggregation(),
#         device: torch.device | None = None,
#     ):
#         super().__init__(
#             inference_wrapper=inference_wrapper,
#             batch_size=batch_size,
#             perturbator=_TokenOcclusionPerturbator(
#                 tokenizer=tokenizer,
#                 inputs_embeddings=inference_wrapper.model.get_input_embeddings(),
#             ),
#             aggregator=aggregator,
#             device=device,
#         )

#     def explain(self, inputs: ModelInput, targets) -> Any:
#         """
#         main process of attribution method
#         """
#         res = []
#         perturbations = self.perturbator.perturb(inputs)
#         for (embeddings, mask), target in zip(perturbations, targets, strict=False):
#             # repeat target tensor to create p axis
#             repeated_target = target.unsqueeze(1).repeat(1, embeddings.shape[1], 1)

#             self.inference_wrapper.to(self.device)
#             results = self.inference_wrapper.batch_inference(embeddings, repeated_target, flatten=True)
#             self.inference_wrapper.cpu()  # TODO: check if we need to do this

#             explanation = self.aggregator(results, mask)
#             res.append(explanation)
#         return res
