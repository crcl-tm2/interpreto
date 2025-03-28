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

from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper


class GenerationInferenceWrapper(InferenceWrapper):
    @singledispatchmethod
    def get_logits(self, model_inputs):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}"
        )

    @get_logits.register
    def _(self, model_inputs: Mapping[str, torch.Tensor]):
        model_inputs = self._embed(model_inputs)
        emb_dim = model_inputs["inputs_embeds"].dim()
        match emb_dim:
            case 2:  # shape: (seq_length, d)
                outputs = self.call_model(**model_inputs)
                return outputs.logits
            case 3:  # shape: (batch, seq_length, d)
                # Chunk along the batch dimension if necessary.
                chunks = torch.split(model_inputs["inputs_embeds"], self.batch_size)
                mask_chunks = torch.split(model_inputs["attention_mask"], self.batch_size)
                logits_list = []
                for chunk, mask_chunk in zip(chunks, mask_chunks, strict=False):
                    chunk_mapping = {"inputs_embeds": chunk, "attention_mask": mask_chunk}
                    outs = self.call_model(**chunk_mapping)
                    logits_list.append(outs.logits)
                return torch.cat(logits_list, dim=0)
            case _:  # Higher dimensions, e.g. (n, p, l, d)
                batch_dims = model_inputs["inputs_embeds"].shape[:-1]
                flat_embeds = model_inputs["inputs_embeds"].flatten(0, -2)
                flat_mask = model_inputs["attention_mask"].flatten(0, -2)
                flat_mapping = {"inputs_embeds": flat_embeds, "attention_mask": flat_mask}
                flat_logits = self.get_logits(flat_mapping)
                return flat_logits.view(*batch_dims, -1)

    @singledispatchmethod
    def get_targeted_logits(self, model_inputs, targets):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targeted_logits in class {self.__class__.__name__}"
        )

    @get_targeted_logits.register
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]):
        model_inputs = self._embed(model_inputs)
        if "input_ids" not in targets:
            raise ValueError("targets must contain 'input_ids' for generative inference.")

        # Get complete logits regardless of the input's shape.
        logits = self.get_logits(model_inputs)
        target_length = targets["input_ids"].shape[-1]

        # For a 2D tensor, assume shape (seq_length, vocab_size)
        if logits.dim() == 2:
            target_logits = logits[-target_length:, :]
            selected_logits = target_logits.gather(dim=-1, index=targets["input_ids"].view(-1, 1)).squeeze(-1)
        else:  # For logits of shape (batch, seq_length, vocab_size) or higher,
            # assume the sequence dimension is the second-to-last.
            target_logits = logits[..., -target_length:, :]
            # For a batch case, unsqueeze the targets so that they match the logits shape.
            selected_logits = target_logits.gather(dim=-1, index=targets["input_ids"].unsqueeze(-1)).squeeze(-1)

        return selected_logits

    @singledispatchmethod
    def get_targets(self, model_inputs, **generation_kwargs):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}"
        )

    @get_targets.register
    def _(self, model_inputs: Mapping[str, torch.Tensor], **generation_kwargs) -> Mapping[str, torch.Tensor]:
        # Generate token IDs using model.generate with additional generation parameters.
        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        # Optionally wrap in a mapping to be consistent.
        return {"input_ids": generated_ids}

    @singledispatchmethod
    def get_gradients(self, model_inputs, targets):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]):
        if "input_ids" not in targets:
            raise ValueError("targets must contain 'input_ids' for gradient computation.")

        # Use the existing _embed function and ensure gradient tracking.
        model_inputs = self._embed(model_inputs)
        inputs_embeds = model_inputs["inputs_embeds"]
        inputs_embeds.requires_grad_(True)

        # Reuse get_logits to compute logits.
        logits = self.get_logits(model_inputs)
        seq_length = logits.shape[0]
        target_length = targets["input_ids"].shape[-1]
        start_idx = seq_length - target_length

        # Preallocate result matrix with NaNs.
        grad_matrix = torch.full((seq_length - 1, target_length), float("nan"), device=logits.device)

        # For each target token, compute gradient of its logit with respect to the input embeddings.
        for j in range(target_length):
            pos = start_idx + j  # index in the full sequence for the j-th target token.
            token_id = targets["input_ids"][j]
            score = logits[pos, token_id]
            # Compute gradient of the selected logit.
            grad = torch.autograd.grad(score, inputs_embeds, retain_graph=True)[0]
            grad_norm = grad.norm(dim=-1)
            # Fill in only those entries corresponding to input tokens preceding the target.
            valid_length = min(pos, seq_length - 1)
            grad_matrix[:valid_length, j] = grad_norm[:valid_length]

        return grad_matrix
