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
    # TODO: maybe add a method basic get_logits and a second get_scores with targets ?
    @singledispatchmethod
    def get_logits(self, model_inputs, targets):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}"
        )

    @get_logits.register
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]):
        model_inputs = self._embed(model_inputs)
        if "inputs_ids" not in targets:
            raise ValueError("targets must contain 'inputs_ids' for generative inference.")

        emb_dim = model_inputs["inputs_embeds"].dim()
        match emb_dim:
            case 2:  # shape: (seq_length, d)
                outputs = self.call_model(**model_inputs)
                logits = outputs.logits  # (seq_length, vocab_size)
                target_length = targets["input_ids"].shape[-1]
                # Assuming target tokens are the last tokens of the sequence.
                target_logits = logits[-target_length:, :]  # shape: (target_length, vocab_size)
                # For each position, select only the logit at the token id provided in targets.
                selected_logits = target_logits.gather(dim=-1, index=targets["input_ids"].view(-1, 1)).squeeze(-1)
                return selected_logits  # shape: (target_length,)
            case 3:  # shape: (batch, seq_length, d)
                # Chunk along the batch dimension if necessary.
                chunks = torch.split(model_inputs["inputs_embeds"], self.batch_size)
                mask_chunks = torch.split(model_inputs["attention_mask"], self.batch_size)
                logits_list = []
                for chunk, mask_chunk in zip(chunks, mask_chunks, strict=False):
                    chunk_mapping = {"inputs_embeds": chunk, "attention_mask": mask_chunk}
                    outs = self.call_model(**chunk_mapping)
                    chunk_logits = outs.logits  # (batch_chunk, seq_length, vocab_size)
                    target_length = targets["input_ids"].shape[-1]
                    # Extract the logits corresponding to the last target_length tokens in each example.
                    chunk_target_logits = chunk_logits[:, -target_length:, :]
                    # For each position, select only the logit at the token id provided in targets.
                    chunk_selected_logits = chunk_target_logits.gather(
                        dim=-1, index=targets["input_ids"].view(-1, 1)
                    ).squeeze(-1)
                    logits_list.append(chunk_selected_logits)
                return torch.cat(logits_list, dim=0)
            case _:  # Higher dimensions, e.g. (n, p, l, d)
                batch_dims = model_inputs["inputs_embeds"].shape[:-1]
                flat_embeds = model_inputs["inputs_embeds"].flatten(0, -2)
                flat_mask = model_inputs["attention_mask"].flatten(0, -2)
                flat_mapping = {"inputs_embeds": flat_embeds, "attention_mask": flat_mask}
                flat_logits = self.get_logits(flat_mapping, targets)
                return flat_logits.view(*batch_dims, -1)

    @singledispatchmethod
    def get_targets(self, model_inputs, **generation_kwargs):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}"
        )

    def _(self, model_inputs: Mapping[str, torch.Tensor], **generation_kwargs) -> Mapping[str, torch.Tensor]:
        # Generate token IDs using model.generate with additional generation parameters.
        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        # Optionally wrap in a mapping to be consistent.
        return {"input_ids": generated_ids}

    # TODO: change get_gradients for generation
    @singledispatchmethod
    def get_gradients(self, model_inputs, targets: torch.Tensor):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: torch.Tensor):
        model_inputs = self._embed(model_inputs)
        scores = self.get_scores(model_inputs, targets)
        scores.backward(torch.ones_like(scores))
        return model_inputs["inputs_embeds"].grad.abs().mean(axis=-1)
