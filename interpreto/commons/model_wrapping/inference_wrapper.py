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

from collections.abc import Generator, Iterator, Mapping, Iterable
from functools import singledispatchmethod

import torch
from transformers import PreTrainedModel

from interpreto.commons.generator_tools import enumerate_generator, PersistentGenerator


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

    def call_model(self, model_inputs: Mapping[str, torch.Tensor]):
        valid_keys = ["input_ids", "input_embeds", "attention_mask"]
        inputs = {key: value.to(self.device) for key, value in model_inputs.items() if key in valid_keys}
        return self.model(**inputs).logits#.cpu()

    @singledispatchmethod
    def get_logits(self, model_inputs):
        raise NotImplementedError(f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}")

    @get_logits.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor]):
        match model_inputs["input_ids"].dim():
            case 1:
                return self.call_model(**model_inputs).logits
            case 2:
                chunks = torch.split(model_inputs["input_ids"], self.batch_size)
                mask_chunks = torch.split(model_inputs["attention_mask"], self.batch_size)
                results = torch.cat([self.call_model({"input_ids":chunk, "attention_mask":mask_chunk}) for chunk, mask_chunk in zip(chunks, mask_chunks)], dim=0)
                return results
            case _:
                batch_dims = model_inputs["input_ids"].shape[:-1]
                flat_input = model_inputs["input_ids"].flatten(0, -2)
                flat_mask = model_inputs["attention_mask"].flatten(0, -2)
                return self.get_logits({"input_ids": flat_input, "attention_mask": flat_mask}).view(*batch_dims, -1)

    @staticmethod
    def _reshape_input_ids(tensor:torch.Tensor):
        # TODO : refaire ça proprement
        match tensor.dim():
            case 1:
                return tensor.unsqueeze(0)
            case 2:
                return tensor
            case 3:
                assert tensor.shape[0] == 1, "When passing a sequence or a generator of inputs to the inference wrapper, please consider giving sequence of perturbations of single elements instead of batches (shape should be (1, n_perturbations, ...))"
                return tensor[0]
            case _:
                raise NotImplementedError(f"input tensor of shape {tensor.shape} not supported")

    @get_logits.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]])->Generator:
        model_inputs = iter(model_inputs)
        # TODO : adapt this method to the "input_embeds" mode
        def concat_and_pad(
            t1: torch.Tensor | None, t2: torch.Tensor, pad_value: int | None = None, dim: int = 0
        ) -> torch.Tensor:
            if t1 is not None:
                if pad_value is not None and t1.shape[1] > t2.shape[1]:
                    t2 = torch.nn.functional.pad(t2, (0, t1.shape[1] - t2.shape[1]), value=pad_value)
                if pad_value is not None and t1.shape[1] < t2.shape[1]:
                    t1 = torch.nn.functional.pad(t1, (0, t2.shape[1] - t1.shape[1]), value=pad_value)
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

        #perturbation_masks = []

        last_item = False

        while True:
            if result_buffer is not None and len(result_indexes) and len(result_buffer) >= result_indexes[0]:
                index = result_indexes.pop(0)
                res = result_buffer[:index]
                result_buffer = result_buffer[index:]
                yield res
                #yield (torch.sum(res * target, dim=-1), perturbation_masks.pop(0))
                if last_item and result_indexes == []:
                    break
                continue
            if last_item or batch is not None and len(batch) == self.batch_size:
                logits = self.call_model({"input_ids":batch, "attention_mask":batch_mask})#self.model(input_ids=batch, attention_mask=batch_mask).logits
                result_buffer = concat_and_pad(result_buffer, logits)
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
                next_item = next(model_inputs)
                input_buffer = self._reshape_input_ids(next_item["input_ids"])
                mask_buffer = self._reshape_input_ids(next_item["attention_mask"])
                result_indexes += [len(input_buffer)]
            except StopIteration:
                if last_item:
                    raise
                last_item = True
        assert len(result_indexes) == len(result_buffer) == len(input_buffer) == len(mask_buffer) == 0, "Some data were not well fetched, please notify it to the developers"

    @singledispatchmethod
    def get_targets(self, model_inputs):
        raise NotImplementedError(f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}")

    @get_targets.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor]) -> Generator:
        return self.get_logits(model_inputs).argmax(dim=-1)

    @get_targets.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]]) -> Iterator:
        yield from (prediction.argmax(dim=-1) for prediction in self.get_logits(iter(model_inputs)))

    def _get_score_from_logits_and_target(self, logits: torch.Tensor, target: torch.Tensor)->torch.Tensor:
        # TODO : change that
        if target.dim() == 1:
            target = target.unsqueeze(0).expand(logits.shape[0], -1)
        return torch.gather(logits, -1, target)

    @singledispatchmethod
    def get_scores(self, model_inputs, targets:torch.Tensor):
        raise NotImplementedError(f"type {type(model_inputs)} not supported for method get_scores in class {self.__class__.__name__}")

    @get_scores.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: torch.Tensor):
        return self._get_score_from_logits_and_target(self.get_logits(model_inputs), targets)

    @get_scores.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor):
        predictions = self.get_logits(iter(model_inputs))
        # TODO : refaire ça proprement
        if targets.dim() in (0, 1):
            targets = targets.view(1, -1)
        if targets.shape[0] == 1:
            for prediction in predictions:
                yield self._get_score_from_logits_and_target(prediction, targets[0:1])
        else:
            for index, prediction in enumerate_generator(predictions):
                print(prediction.shape, targets[index].shape)
                # TODO : refaire ça proprement
                yield self._get_score_from_logits_and_target(prediction, targets[index:index+1])

    @singledispatchmethod
    def get_gradients(self, model_inputs, targets: torch.Tensor):
        raise NotImplementedError(f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}")

    @get_gradients.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: torch.Tensor):
        scores = self.get_scores(model_inputs, targets)
        scores.backward(torch.ones_like(scores))
        return model_inputs["input_ids"].grad

    @get_gradients.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor):
        # TODO : see if we can do that without using persistent generator
        mi = PersistentGenerator(iter(model_inputs))
        scores = self.get_scores(mi, targets)
        for index, element in enumerate_generator(scores):
            element.backward(torch.ones_like(element))
            return mi[index]["input_ids"].grad
