from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterable
from functools import singledispatchmethod
import torch
from nnsight import NNsight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from interpreto.attributions.aggregations.base import MaskwiseMeanAggregation
from interpreto.commons.model_wrapping.inference_wrapper import HuggingFaceClassifierWrapper
from interpreto.attributions.base import InferenceExplainer

ModelInput = str | Iterable | torch.Tensor | Mapping[str, torch.Tensor] 

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

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

    def perturb_ids(self, tokens:Mapping) -> Mapping[str, torch.Tensor]:
        # Default implementation, should be overriden
        return tokens

    def perturb_tensors(self, tensors:torch.Tensor) -> torch.Tensor:
        # Default implementation, should be overriden
        raise NotImplementedError(f"No way to perturb embeddings has been defined in {self.__class__.__name__}")

class MaskBasedPerturbator(BasePerturbator):
    def get_mask(self, *args, **kwargs):
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
                        inputs_embeddings:torch.nn.Module|None=None,
                        n_perturbations:int=1,
                        granularity_level:str="token"):
        super().__init__(tokenizer=tokenizer, inputs_embedder=inputs_embeddings)
        self.n_perturbations = n_perturbations
        self.granularity_level = granularity_level

    def get_theorical_masks(self, sizes:torch.Tensor) -> torch.Tensor:
        # input : 1d tensor containing n values of l_spec where :
        # l_spec = length of the sequence according to certain granularity level (nb words, nb tokens, nb sentences, etc.)
        # n = batch size
        p = self.n_perturbations
        # TODO : redefine this
        return [torch.randbool(p, l).long() for l in sizes]
    
    def get_practical_mask_from_theorical_mask(self, mask:torch.Tensor, granularity_matrix:torch.Tensor) -> torch.Tensor:
        # mask : T-mask furnished by get_theorical_masks
        # granularity_matrix : for editable tokens to tokens : matrix of editable tokens (offset_mapping.sum(dim=-1).bool().long())
        granularity_function = ...

    def get_mask(self, tokens:Mapping) -> torch.Tensor:
        nl_matrix = ...
        return self.get_practical_mask_from_theorical_mask(self.get_theorical_masks(nl_matrix), ...)


    def perturb_tokens(self, tokens:Mapping) -> dict[str, torch.Tensor]:
        #tokens["perturbation_mask"] = self.get_mask(tokens) * tokens["offset_mapping"].sum(dim=-1).bool().long().unsqueeze(-1)
        tokens["perturbation_mask"] = self.get_mask(tokens)

        tokens["inputs_ids"] = self.apply_mask(
            inputs=tokens["input_ids"].unsqueeze(-1),
            mask=tokens["perturbation_mask"],
            mask_value=torch.Tensor([self.default_mask_id])
        )
        return tokens

class EmbeddingsMaskBasedPerturbator(MaskBasedPerturbator):
    def default_mask_vector(self):
        return self.inputs_embedder.weight[self.default_mask_id]

    def perturb_embeddings(self, embeddings):
        mask = self.get_mask(embeddings)
        ...
        perturbed_embeddings = self.apply_mask(embeddings, mask, self.default_mask_vector)
        return {"perturbation_mask":mask, "inputs_embeds":perturbed_embeddings}

class TokenOcclusionPerturbator(TokenMaskBasedPerturbator):
    def get_mask(self, tokens:Mapping) -> torch.Tensor:
        # TODO : dégager ça
        perturbable_mask = tokens["offset_mapping"].sum(dim=-1).bool().long()
        attention_masks = tokens["offset_mapping"]
        max_attention_mask_length = attention_masks.sum(dim=1).max().item()
        return torch.diag_embed(perturbable_mask)[:, :max_attention_mask_length]

class PerturbatorChain(BasePerturbator):
    ...

class PerturbationPool(BasePerturbator):
    ...

my_pert = TokenOcclusionPerturbator(tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME))

sentences = my_pert.perturb(["Hello, how are you?"])
print(sentences)