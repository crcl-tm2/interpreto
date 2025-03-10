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

import functools
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from nnsight import NNsight
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedModel

class BaseInferenceWrapper:
    def __init__(self, model:PreTrainedModel,
                 batch_size:int,
                 device:torch.device|None=None):
        self.model = model
        self.model.to(device or torch.device("cpu"))
        self.batch_size = batch_size

        self.__buffer_indexes = []
        self.__buffer = torch.empty(0, 0, 0)
        self.__rest = None
        self.__rest_mask = None

    @property
    def __n_finished(self):
        return len(self.__buffer_indexes)

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device:torch.device):
        self.model.to(device)

    def to(self, device:torch.device):
        self.device = device

    def cpu(self):
        self.device = torch.device("cpu")

    def cuda(self):
        self.device = torch.device("cuda")

    def call_model(self, model_inputs:Mapping[str,torch.Tensor]) -> ModelOutput:
        print("calling model !")
        print(model_inputs)
        print("input batch size : ", model_inputs["input_ids"].shape[0], " required batch size : ", self.batch_size)
        return self.model(**model_inputs)

    def inference(self, inputs:Iterable[Mapping[str,torch.Tensor]]) -> torch.Tensor:

        # TODO : generalize to other tokenizers
        pad_token_id = 0

        while True:
            while not self.__n_finished:
                if self.__rest is not None:
                    p = self.__rest.shape[1]
                    batchs = self.__rest.split(self.batch_size)
                    batch_masks = self.__rest_mask.split(self.batch_size)
                    self.__rest = None
                    self.__rest_mask = None
                else:
                    # TODO : handle stop iteration
                    try:
                        item = next(inputs)
                    except StopIteration:
                        # check that everything is finished
                        #TODO : remove and raise
                        assert self.__buffer.shape[0] == 0
                        assert self.__rest is None
                        raise
                    # TODO : allow inputs_embeds
                    # TODO : use attention mask and give it to the model
                    assert item["input_ids"].shape[0] == 1, "WTF ?"

                    input_ids = item["input_ids"][0]
                    p = input_ids.shape[0]
                    # TODO : do this mask repetition in the perturbator
                    att_mask = item["attention_mask"].repeat(p, 1)
                    # TODO : handle case where no attention mask
                    batchs = list(input_ids.split(self.batch_size))
                    batch_masks = list(att_mask.split(self.batch_size))
                new_indexes = [self.__buffer.shape[0] + p]
                while len(batchs[-1]) < self.batch_size:
                    try:
                        other_item = next(inputs)
                        other_input_ids = item["input_ids"][0]
                        other_att_mask = item["attention_mask"].repeat(other_input_ids.shape[0], 1)
                    except StopIteration:
                        break
                    
                    
                    # TODO : do this mask repetition in the perturbator
                    att_mask = item["attention_mask"].repeat(p, 1)

                    other_item_length = len(other_input_ids)
                    missing_length = self.batch_size - len(batchs[-1])

                    # padding of shorter sequences
                    batch_seq_length = batchs[-1].shape[1]
                    other_item_seq_length = other_input_ids.shape[1]
                    if other_item_seq_length != batch_seq_length:
                        pad = torch.full((batchs[-1].shape[0], other_item_seq_length - batch_seq_length), pad_token_id)
                        batchs[-1] = torch.cat([batchs[-1], pad], dim=-1)
                        batch_masks[-1] = torch.cat([batch_masks[-1], torch.zeros_like(pad)], dim=-1)
                    batchs[-1] = torch.cat([batchs[-1], other_input_ids[:missing_length]])
                    batch_masks[-1] = torch.cat([batch_masks[-1], other_att_mask[:missing_length]])
                    if other_item_length > missing_length:
                        self.__rest = other_input_ids[missing_length:]
                    else:
                        new_indexes += [new_indexes[-1] + other_item_length]
                for batch, attention_mask in zip(batchs, batch_masks):
                    model_inputs = {"input_ids":batch, "attention_mask":attention_mask}
                    result = self.call_model(model_inputs).logits
                    self.__buffer = torch.cat([self.__buffer, result], dim=0)
                self.__buffer_indexes += new_indexes
            index = self.__buffer_indexes.pop(0)
            res = self.__buffer[:index]
            self.__buffer = self.__buffer[index:]
            yield res



class ClassificationInferenceWrapper:
    def __init__(self, model: NNsight | torch.nn.Module, batch_size: int, device: torch.device | None = None):
        if isinstance(model, torch.nn.Module):
            model = NNsight(model)
        self.model = model
        self.model.to(device)
        assert batch_size > 0, "Batch size must be a positive integer."
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cpu")

    def to(self, device: torch.device):
        self.model.to(device)

    def cpu(self):
        self.model.cpu()

    def cuda(self):
        self.model.cuda()

    @staticmethod
    def flatten_unflatten(func: Callable) -> Callable:
        """
        A decorator that flattens multiple batch dimensions before calling the function
        and unflattens the output back to the original shape.

        It introduces a 'flatten' argument to control this behavior.

        Args:
            func (Callable): The function to wrap.

        Returns:
            Callable: The wrapped function.
        """

        @functools.wraps(func)
        def wrapper(
            self, inputs: torch.Tensor, target: torch.Tensor, *args: Any, flatten: bool = False, **kwargs: Any
        ) -> torch.Tensor:
            """
            Wrapper that flattens and unflattens the inputs tensor based on the 'flatten' flag.

            Args:
                inputs (torch.Tensor): Inputs tensor of shape (n, p, ...).
                flatten (bool): Whether to flatten before and unflatten after.

            Returns:
                torch.Tensor: Processed tensor with restored shape if flatten=True.
            """
            if not isinstance(inputs, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("Expected 'inputs' and 'targets' to be a PyTorch tensor.")

            dims_to_flatten = inputs.shape[:2]  # Store original shape

            # Flatten if requested
            if flatten:
                inputs = inputs.flatten(start_dim=0, end_dim=1)  # Shape: (n*p, ...)
                target = target.flatten(start_dim=0, end_dim=1)  # Shape: (n*p, ...)

            # Call the original function
            outputs = func(self, inputs, target, *args, **kwargs)

            # Unflatten if needed
            if flatten and isinstance(outputs, torch.Tensor):
                outputs = outputs.unflatten(dim=0, sizes=dims_to_flatten)  # Restore shape: (n, p, output_dim)
            return outputs

        return wrapper

    # Temporary
    # TODO : eventually deal with that in a better way (automatic model wrapping or decorating ?)
    def call_model(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs.to(self.device))

    def inference(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.call_model(inputs)

        return torch.sum(logits * target, dim=-1)

    def gradients(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clone().detach().requires_grad_(True)  # TODO: verify the clone, not sure useful
        scores = self.inference(inputs, target)
        scores.backward(torch.ones_like(scores))  # Allow multiple sample dimensions (n, p)
        return inputs.grad

    @flatten_unflatten
    def batch_inference(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scores = []
        for i in range(0, inputs.shape[0], self.batch_size):
            batch = inputs[i : i + self.batch_size].to(self.device)
            target = targets[i : i + self.batch_size].to(self.device)
            batch_scores = self.inference(batch, target).cpu()
            scores.append(batch_scores)
        return torch.cat(scores, dim=0)

    @flatten_unflatten
    def batch_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gradients = []
        for i in range(0, inputs.shape[0], self.batch_size):
            batch = inputs[i : i + self.batch_size].to(self.device)
            target = targets[i : i + self.batch_size].to(self.device)
            batch_gradients = self.gradients(batch, target).detach().cpu()
            gradients.append(batch_gradients)
        return torch.cat(gradients, dim=0)


class HuggingFaceClassifierWrapper(ClassificationInferenceWrapper):
    def call_model(self, inputs: torch.Tensor) -> ModelOutput:
        # TODO : deal with cases where logits is in "start_logits" or "end_logit" attributes
        return self.model(inputs_embeds=inputs).logits
