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

from collections.abc import Generator, Iterable, Mapping

import torch
from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput


class ClassificationInferenceWrapper:
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

    def call_model(self, model_inputs: Mapping[str, torch.Tensor]) -> ModelOutput:
        return self.model(**model_inputs).logits

    def inference(self, inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor) -> Generator:
        def concat_and_pad(
            t1: torch.Tensor | None, t2: torch.Tensor, pad_value: int | None = None, dim: int = 0
        ) -> torch.Tensor:
            if t1 is not None:
                if pad_value is not None and t1.shape[1] > t2.shape[1]:
                    t2 = torch.nn.functional.pad(t2, (0, t1.shape[1] - t2.shape[1]), value=pad_value)
                return torch.cat([t1, t2], dim=dim)
            return t2

        # TODO : Check that tokenizers other than bert also use 0 as pad_token_id, otherwise, fix this
        pad_token_id = 0

        result_buffer: torch.Tensor | None = None
        result_indexes: list[int] = []

        batch: torch.Tensor | None = None
        batch_mask: torch.Tensor | None = None

        input_buffer = torch.zeros(0)
        mask_buffer = torch.zeros(0)

        perturbation_masks = []

        last_item = False

        while True:
            if result_buffer is not None and len(result_buffer) >= result_indexes[0]:
                index = result_indexes.pop(0)
                res = result_buffer[:index]

                target = targets[0].unsqueeze(0).repeat(res.shape[0], 1)
                targets = targets[1:]

                result_buffer = result_buffer[index:]

                yield (torch.sum(res * target, dim=-1), perturbation_masks.pop(0))
                if last_item:
                    break
                continue
            if last_item or batch is not None and len(batch) == self.batch_size:
                exec_result = self.call_model({"input_ids": batch, "attention_mask": batch_mask})
                result_buffer = concat_and_pad(result_buffer, exec_result)
                batch = batch_mask = None
                continue
            if input_buffer.numel():
                missing_length = self.batch_size - len(batch if batch is not None else ())
                batch = concat_and_pad(batch, input_buffer[:missing_length], pad_token_id)
                batch_mask = concat_and_pad(batch_mask, mask_buffer[:missing_length], 0)
                input_buffer = input_buffer[missing_length:]
                mask_buffer = mask_buffer[missing_length:]
                continue
            try:
                next_item, perturbation_mask = next(inputs)
                perturbation_masks.append(perturbation_mask)
                input_buffer = next_item["input_ids"][0]
                mask_buffer = next_item["attention_mask"][0]

                result_indexes += [len(input_buffer)]
            except StopIteration:
                last_item = True

    def gradients(self, inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor) -> Generator:
        # TODO : check
        results = self.inference(inputs, targets)
        for result, mask in results:
            result.backward(torch.ones_like(result))
            yield result.grad, mask
