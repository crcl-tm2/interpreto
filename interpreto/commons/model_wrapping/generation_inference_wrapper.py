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

from __future__ import annotations

from collections.abc import Mapping
from functools import singledispatchmethod

import torch
from torch.autograd.functional import jacobian

from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper


class GenerationInferenceWrapper(InferenceWrapper):
    @singledispatchmethod
    def get_logits(self, model_inputs):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}"
        )

    @get_logits.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor]):
        model_inputs = self._embed(model_inputs)
        emb_dim = model_inputs["inputs_embeds"].dim()
        match emb_dim:
            case 2:  # shape: (l, d)
                outputs = self.call_model(model_inputs)
                return outputs.logits  # (l, v)
            case 3:  # shape: (n, l, d)
                # Chunk along the batch dimension if necessary.
                chunks = torch.split(model_inputs["inputs_embeds"], self.batch_size)
                mask_chunks = torch.split(model_inputs["attention_mask"], self.batch_size)
                logits_list = []
                for chunk, mask_chunk in zip(chunks, mask_chunks, strict=False):
                    chunk_mapping = {"inputs_embeds": chunk, "attention_mask": mask_chunk}
                    outs = self.call_model(chunk_mapping)
                    logits_list.append(outs.logits)
                return torch.cat(logits_list, dim=0)  # (n, l, v)
            case _:  # Higher dimensions, e.g. (n, p, l, d)
                batch_dims = model_inputs["inputs_embeds"].shape[:-1]
                flat_embeds = model_inputs["inputs_embeds"].flatten(start_dim=0, end_dim=-3)  # (n * p, l, d)
                flat_mask = model_inputs["attention_mask"].flatten(start_dim=0, end_dim=-3)  # (n * p, l, d)
                flat_mapping = {"inputs_embeds": flat_embeds, "attention_mask": flat_mask}
                flat_logits = self.get_logits(flat_mapping)
                return flat_logits.view(*batch_dims, -1)  # (n, p, l, v)

    @singledispatchmethod
    def get_targeted_logits(self, model_inputs, targets):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targeted_logits in class {self.__class__.__name__}"
        )

    @get_targeted_logits.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]):
        # remove last target token from the model inputs
        # to avoid using the last token in the generation process
        model_inputs = {
            key: value[..., :-1, :] if key == "inputs_embeds" else value[..., :-1]
            for key, value in model_inputs.items()
        }
        model_inputs = self._embed(model_inputs)
        if "input_ids" not in targets:
            raise ValueError("targets must contain 'input_ids' for generative inference.")
        # Get complete logits regardless of the input's shape.
        logits = self.get_logits(model_inputs)  # (l-1, v) | (n, l-1, v) | (n, p, l-1, v)
        target_length = targets["input_ids"].shape[-1]  # lt < l

        # assume the sequence dimension is the second-to-last.
        target_logits = logits[..., -target_length:, :]

        if targets["input_ids"].shape != target_logits.shape[:-1]:
            raise ValueError(
                "target logits shape without the vocabulary dimension must match the target inputs ids shape."
                f"Got {target_logits.shape[:-1]} and {targets['input_ids'].shape}."
            )

        # For a batch case, unsqueeze the targets so that they match the logits shape.
        selected_logits = target_logits.gather(dim=-1, index=targets["input_ids"].unsqueeze(-1)).squeeze(-1)

        return selected_logits

    @singledispatchmethod
    def get_targets(self, model_inputs, **generation_kwargs):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}"
        )

    @get_targets.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], **generation_kwargs) -> Mapping[str, torch.Tensor]:
        # Generate token IDs using model.generate with additional generation parameters.
        full_ids = self.model.generate(**model_inputs, **generation_kwargs)
        original_length = model_inputs["attention_mask"].shape[-1]
        generated_ids = full_ids[..., original_length:]
        generated_attention_mask = torch.ones_like(generated_ids)
        full_attention_mask = torch.cat([model_inputs["attention_mask"], generated_attention_mask], dim=-1)
        full_mapping = {"input_ids": full_ids, "attention_mask": full_attention_mask}
        generated_mapping = {"input_ids": generated_ids, "attention_mask": generated_attention_mask}
        return full_mapping, generated_mapping

    @singledispatchmethod
    def get_gradients(self, model_inputs, targets):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]):
        model_inputs = self._embed(model_inputs)
        inputs_embeds = model_inputs["inputs_embeds"]

        def get_score(inputs_embeds: torch.Tensor):
            return self.get_targeted_logits(
                {"inputs_embeds": inputs_embeds, "attention_mask": model_inputs["attention_mask"]}, targets
            )

        # Compute gradient of the selected logits:
        grad_matrix = jacobian(get_score, inputs_embeds)  # (n, lt, n, l, d)
        grad_matrix = grad_matrix.mean(dim=-1)  # (n, lt, n, l)  # average over the embedding dimension
        n = grad_matrix.shape[0]  # number of examples in the batch
        diag_grad_matrix = grad_matrix[
            torch.arange(n), :, torch.arange(n), :
        ]  # (n, lt, l) # remove the gradients of the other examples

        return diag_grad_matrix
