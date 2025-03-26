from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Mapping
from functools import singledispatchmethod

import torch
from transformers import PreTrainedModel

from interpreto.commons.generator_tools import enumerate_generator


class GenerationInferenceWrapper:
    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int,
        device: torch.device | None = None,
        generation_kwargs: dict | None = None,
    ):
        """
        :param model: Generative model (e.g. GPT, LLaMA, Gemma, ...)
        :param batch_size: Maximum number of prompts processed simultaneously
        :param device: Device on which to run the model (CPU or GPU)
        :param generation_kwargs: Default parameters to be passed to generate (e.g. max_length, do_sample, etc.)
        """
        self.model = model
        self.model.to(device or torch.device("cpu"))
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs or {}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @device.setter
    def device(self, device: torch.device):
        self.model.to(device)

    def to(self, device: torch.device):
        self.device = device

    def cpu(self):
        self.device = torch.device("cpu")

    def cuda(self):
        self.device = torch.device("cuda")

    def call_model(self, model_inputs: Mapping[str, torch.Tensor], **gen_kwargs):
        """
        Calls the model's generate method with filtered inputs and generate parameters.
        """
        valid_keys = ["input_ids", "attention_mask", "input_embeds"]
        inputs = {k: v.to(self.device) for k, v in model_inputs.items() if k in valid_keys}
        kwargs = {**self.generation_kwargs, **gen_kwargs}
        return self.model.generate(**inputs, **kwargs)

    @singledispatchmethod
    def get_targets(self, model_inputs, **gen_kwargs):
        """Generate targets (text) from model inputs"""
        raise NotImplementedError(f"Type {type(model_inputs)} not supported for method get_targets")

    @get_targets.register
    def _(self, model_inputs: Mapping[str, torch.Tensor], **gen_kwargs):
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("The mapping must contain the key 'input_ids'.")
        if input_ids.dim() == 1:
            batch_mapping = {k: v.unsqueeze(0) for k, v in model_inputs.items()}
            generated = self.call_model(batch_mapping, **gen_kwargs)
            return generated[0]
        elif input_ids.dim() == 2:
            if input_ids.size(0) > self.batch_size:
                chunks = torch.split(input_ids, self.batch_size, dim=0)
                if "attention_mask" in model_inputs:
                    mask_chunks = torch.split(model_inputs["attention_mask"], self.batch_size, dim=0)
                else:
                    mask_chunks = [None] * len(chunks)
                outputs = []
                for i, chunk in enumerate(chunks):
                    chunk_mapping = {"input_ids": chunk}
                    if mask_chunks[i] is not None:
                        chunk_mapping["attention_mask"] = mask_chunks[i]
                    outputs.append(self.call_model(chunk_mapping, **gen_kwargs))
                return torch.cat(outputs, dim=0)
            else:
                return self.call_model(model_inputs, **gen_kwargs)
        else:
            batch_dims = input_ids.shape[:-1]
            flat_input = input_ids.flatten(0, -2)
            new_mapping = dict(model_inputs)
            new_mapping["input_ids"] = flat_input
            if "attention_mask" in model_inputs:
                new_mapping["attention_mask"] = model_inputs["attention_mask"].flatten(0, -2)
            flat_generated = self.get_targets(new_mapping, **gen_kwargs)
            return flat_generated.view(*batch_dims, -1)

    @get_targets.register
    def _(self, model_inputs: Iterator[Mapping[str, torch.Tensor]], **gen_kwargs) -> Generator:
        batch_mappings = []
        for mapping in model_inputs:
            batch_mappings.append(mapping)
            if len(batch_mappings) == self.batch_size:
                combined = {}
                for key in batch_mappings[0]:
                    combined[key] = torch.cat([m[key] for m in batch_mappings], dim=0)
                generated = self.get_targets(combined, **gen_kwargs)
                for i in range(generated.size(0)):
                    yield generated[i]
                batch_mappings = []
        if batch_mappings:
            combined = {}
            for key in batch_mappings[0]:
                combined[key] = torch.cat([m[key] for m in batch_mappings], dim=0)
            generated = self.get_targets(combined, **gen_kwargs)
            for i in range(generated.size(0)):
                yield generated[i]

    @get_targets.register
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]], **gen_kwargs) -> Generator:
        return self.get_targets(iter(model_inputs), **gen_kwargs)

    @singledispatchmethod
    def get_logits(self, model_inputs, targets: Mapping[str, torch.Tensor], **gen_kwargs):
        raise NotImplementedError(f"Type {type(model_inputs)} non supporté pour get_logits")

    @get_logits.register
    def _(
        self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], **gen_kwargs
    ) -> torch.Tensor:
        """
        Calcule les logits pour chaque token de la target en mode teacher forcing.
        Pour un prompt et une target donnés, on construit itérativement le contexte (prompt + tokens déjà ajoutés)
        et on réalise un forward pass pour obtenir le logit du token suivant.
        Retourne un tenseur de forme (target_length, batch_size, vocab_size) pour un batch,
        ou (target_length, vocab_size) pour un prompt unique.
        """
        if "input_ids" not in targets:
            raise ValueError("Le mapping target doit contenir la clé 'input_ids'")
        targets_tensor = targets["input_ids"]

        if "input_embeds" not in model_inputs:
            if "input_ids" not in model_inputs:
                raise ValueError("Le mapping model_inputs doit contenir 'input_embeds' ou 'input_ids'")
            input_embeds = self.model.get_input_embeddings()(model_inputs["input_ids"])
        else:
            input_embeds = model_inputs["input_embeds"]

        batch_size = input_embeds.size(0)
        target_length = targets_tensor.size(1)
        logits_list = []
        for t in range(target_length):
            input_embeds.requires_grad_(True)
            outputs = self.model(input_embeds=input_embeds, use_cache=False)
            logits_t = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            logits_list.append(logits_t)
            next_token = targets_tensor[:, t].unsqueeze(1)  # (batch_size, 1)

        logits_tensor = torch.stack(logits_list, dim=0)  # (target_length, batch_size, vocab_size)
        if logits_tensor.size(1) == 1:
            logits_tensor = logits_tensor.squeeze(1)  # (target_length, vocab_size)
        return logits_tensor

    @get_logits.register
    def _(
        self, model_inputs: Iterator[Mapping[str, torch.Tensor]], targets: Mapping[str, torch.Tensor], **gen_kwargs
    ) -> Generator:
        for index, mapping in enumerate_generator(model_inputs):
            yield self.get_logits(mapping, targets, **gen_kwargs)

    @get_logits.register
    def _(
        self, model_inputs: Iterable[Mapping[str, torch.Tensor]], targets: Mapping[str, torch.Tensor], **gen_kwargs
    ) -> Generator:
        return self.get_logits(iter(model_inputs), targets, **gen_kwargs)

    @singledispatchmethod
    def get_gradients(self, model_inputs, targets: Mapping[str, torch.Tensor], **gen_kwargs):
        raise NotImplementedError(f"Type {type(model_inputs)} non supporté pour get_gradients")

    @get_gradients.register
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], **gen_kwargs) -> list:
        """
        Calcule les gradients pour chaque token de la target en mode teacher forcing.
        Pour chaque pas t, on construit le contexte (prompt + tokens déjà traités), on effectue un forward pass
        pour obtenir le logit du token cible, puis on lance backward pour obtenir le gradient par rapport aux embeddings.
        Renvoie une liste contenant, pour chaque token, le tenseur de gradient.
        """
        if "input_ids" not in targets:
            raise ValueError("Le mapping target doit contenir la clé 'input_ids'")
        targets_tensor = targets["input_ids"]

        if "input_ids" not in model_inputs:
            raise ValueError("Le mapping model_inputs doit contenir 'input_ids'")
        prompt_ids = model_inputs["input_ids"]
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if targets_tensor.dim() == 1:
            targets_tensor = targets_tensor.unsqueeze(0)
        batch_size = prompt_ids.size(0)
        target_length = targets_tensor.size(1)
        gradients_list = []
        context_ids = prompt_ids.clone()
        for t in range(target_length):
            input_embeds = self.model.get_input_embeddings()(context_ids)
            input_embeds.requires_grad_(True)
            outputs = self.model(input_embeds=input_embeds, use_cache=False)
            logits_t = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            target_token = targets_tensor[:, t].unsqueeze(1)  # (batch_size, 1)
            score_t = torch.gather(logits_t, dim=1, index=target_token)  # (batch_size, 1)
            self.model.zero_grad()
            if input_embeds.grad is not None:
                input_embeds.grad.zero_()
            score_t.backward(retain_graph=True)
            grad_t = input_embeds.grad.detach().clone()
            gradients_list.append(grad_t)
            context_ids = torch.cat([context_ids, target_token], dim=1)
        return gradients_list

    @get_gradients.register
    def _(
        self, model_inputs: Iterator[Mapping[str, torch.Tensor]], targets: Mapping[str, torch.Tensor], **gen_kwargs
    ) -> Generator:
        for index, mapping in enumerate_generator(model_inputs):
            yield self.get_gradients(mapping, targets, **gen_kwargs)

    @get_gradients.register
    def _(
        self, model_inputs: Iterable[Mapping[str, torch.Tensor]], targets: Mapping[str, torch.Tensor], **gen_kwargs
    ) -> Generator:
        return self.get_gradients(iter(model_inputs), targets, **gen_kwargs)
