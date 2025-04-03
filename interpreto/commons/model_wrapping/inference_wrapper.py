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

from typing import Any
import warnings
from collections.abc import Iterable, Mapping, Generator
from functools import singledispatchmethod

import torch
from transformers import PreTrainedModel



#TODO : move that somewhere else
def concat_and_pad(
    *tensors: torch.Tensor | None,
    pad_left : bool,
    dim: int = 0,
    pad_value: int = 0,
    pad_dims: Iterable[int] | None = None,
) -> torch.Tensor:
    _tensors = [a for a in tensors if a is not None]
    tensors_dim = _tensors[0].dim() #should be equal for all tensors, otherwise raise an error
    pad_dims = pad_dims or []
    max_length_per_dim = [max(t.shape[d] for t in _tensors) for d in pad_dims]

    res = []
    for t in _tensors:
        pad = [0, 0] * tensors_dim
        for pad_dim, pad_length in zip(pad_dims, max_length_per_dim, strict=True):
            pad_index = -2 * (pad_dim % tensors_dim) - 1 - pad_left
            pad[pad_index] = pad_length - t.shape[pad_dim]
        res += [torch.nn.functional.pad(t, pad, value=pad_value)]
    return torch.cat(res, dim=dim)

class InferenceWrapper:
    PAD_LEFT = True

    def __init__(self, model: PreTrainedModel, batch_size: int, device: torch.device | None = None):
        self.model = model
        self.model.to(device or torch.device("cpu"))
        self.batch_size = batch_size

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device: torch.device):
        self.model.to(device)

    def to(self, device: torch.device):
        self.device = device

    def cpu(self):
        self.device = torch.device("cpu")

    def cuda(self):
        self.device = torch.device("cuda")

    def _embed(self, model_inputs: Mapping[str, torch.Tensor]):
        if "inputs_embeds" in model_inputs:
            return model_inputs
        if "input_ids" in model_inputs:
            # TODO : flatten/unflatten
            model_inputs["inputs_embeds"] = self.model.get_input_embeddings()(model_inputs.pop("input_ids"))
            return model_inputs
        raise ValueError("model_inputs should contain either 'input_ids' or 'inputs_embeds'")

    # def call_model(self, model_inputs: Mapping[str, torch.Tensor]):
    #     valid_keys = ["inputs_embeds", "input_ids", "attention_mask"]
    #     inputs = {key: value.to(self.device) for key, value in model_inputs.items() if key in valid_keys}
    #     for k, v in inputs.items():
    #         if v.shape[0] > self.batch_size:
    #             warnings.warn(
    #                 f"Batch size of {k} ({model_inputs.get(k).shape[0]}) is greater than the wrapper's batch size of {self.batch_size}. "
    #                 f"Consider adjust the batch size or the wrapper of split your data."
    #             )
    #             break
    #     return self.model(**inputs)

    def call_model(self, input_embeds:torch.Tensor, attention_mask:torch.Tensor):
        input_embeds.to(self.device)
        attention_mask.to(self.device)
        if input_embeds.shape[0] > self.batch_size:
            warnings.warn(
                f"Batch size of {input_embeds.shape[0]} is greater than the wrapper's batch size of {self.batch_size}. "
                f"Consider adjust the batch size or the wrapper of split your data.",
                stacklevel=1
            )
        return self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    
    @singledispatchmethod
    def get_logits(self, model_inputs:Any)->torch.Tensor|Generator[torch.Tensor, None, None]:
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}"
        )

    @get_logits.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor])->torch.Tensor:
        model_inputs = self._embed(model_inputs)
        match model_inputs["inputs_embeds"].dim():
            case 2:  # (l, d)
                return self.call_model(**model_inputs).logits
            case 3:  # (p, l, d) or (n, l, d)
                embeds_chunks = model_inputs["inputs_embeds"].split(self.batch_size)
                mask_chunks = model_inputs["attention_mask"].split(self.batch_size)
                return torch.cat(
                    [
                        self.call_model(embeds_chunk, mask_chunk).logits
                        for embeds_chunk, mask_chunk in zip(embeds_chunks, mask_chunks, strict=False)
                    ],
                )
            case _:  # (..., l, d) like (n, p, l, d)
                return self.get_logits({k:v.flatten(0, -3) for k, v in model_inputs.items()}).view(
                    *model_inputs["inputs_embeds"].shape[:-1], -1
                )

    def _reshape_inputs(self, tensor: torch.Tensor, non_batch_dims: int = 2)->torch.Tensor:
        # TODO : refaire ça proprement
        assert tensor.dim() >= non_batch_dims, (
            "The given tensor have less dimensions than non_batch_dims parameter"
        )
        if tensor.dim() == non_batch_dims:
            return tensor.unsqueeze(0)
        if tensor.dim() == non_batch_dims + 1:
            return tensor
        assert tensor.shape[0] == 1, (
            "When passing a sequence or a generator of inputs to the inference wrapper, please consider giving sequence of perturbations of single elements instead of batches (shape should be (1, n_perturbations, ...))"
        )
        return self._reshape_inputs(tensor[0], non_batch_dims=non_batch_dims)

    @get_logits.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]]) -> Generator[torch.Tensor, None, None]:

        model_inputs = iter(model_inputs)

        # TODO : Check that tokenizers other than bert also use 0 as pad_token_id, otherwise, fix this
        pad_token_id = 0

        result_buffer: torch.Tensor | None = None
        result_indexes: list[int] = []

        batch: torch.Tensor | None = None
        batch_mask: torch.Tensor | None = None

        input_buffer = torch.zeros(0)
        mask_buffer = torch.zeros(0)

        last_item = False

        while True:
            if result_buffer is not None and len(result_indexes) and len(result_buffer) >= result_indexes[0]:
                index = result_indexes.pop(0)
                yield result_buffer[:index]
                result_buffer = result_buffer[index:]
                if last_item and not result_indexes:
                    break
                continue
            if last_item or batch is not None and len(batch) == self.batch_size:
                logits = self.call_model(batch, batch_mask).logits
                result_buffer = concat_and_pad(result_buffer, logits, pad_left=self.PAD_LEFT)
                batch = batch_mask = None
                continue
            if input_buffer.numel():
                missing_length = self.batch_size - len(batch if batch is not None else ())
                batch = concat_and_pad(batch, input_buffer[:missing_length], pad_left=self.PAD_LEFT, dim=0, pad_value=pad_token_id, pad_dims=(1,))
                batch_mask = concat_and_pad(batch_mask, mask_buffer[:missing_length], pad_left=self.PAD_LEFT, dim=0, pad_value=0, pad_dims=(-1,))
                input_buffer = input_buffer[missing_length:]
                mask_buffer = mask_buffer[missing_length:]
                continue
            try:
                next_item = self._embed(next(model_inputs))
                input_buffer = self._reshape_inputs(next_item["inputs_embeds"], non_batch_dims=2)
                mask_buffer = self._reshape_inputs(next_item["attention_mask"], non_batch_dims=1)
                result_indexes += [len(input_buffer)]
            except StopIteration:
                if last_item:
                    raise
                last_item = True
        if any(len(a) for a in [result_indexes, result_buffer, input_buffer, mask_buffer]):
            warnings.warn("Some data were not well fetched in inference wrapper, please check your code if you made custom method or notify it to the developers")
