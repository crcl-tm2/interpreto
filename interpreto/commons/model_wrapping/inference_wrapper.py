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
from collections.abc import Callable
from typing import Any

import torch
from nnsight import NNsight
from transformers.utils.generic import ModelOutput


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
