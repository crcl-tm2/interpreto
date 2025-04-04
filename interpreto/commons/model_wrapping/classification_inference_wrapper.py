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

from collections.abc import Iterable, Iterator, Mapping
from functools import singledispatchmethod

import torch

from interpreto.commons.generator_tools import PersistentGenerator, enumerate_generator
from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper


class ClassificationInferenceWrapper(InferenceWrapper):
    PAD_LEFT = False
    """
        Basic inference wrapper for classification tasks.
    """

    def _process_target(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # TODO : refaire ça proprement
        match target.dim():
            case 0:
                return self._process_target(target.unsqueeze(0), logits)
            case 1:  # (t)
                index_shape = list(logits.shape)
                index_shape[-1] = target.shape[0]
                return target.expand(index_shape)
            case 2:  # (n, t)
                if target.shape[0] == 1:
                    return target.expand(logits.shape[0], -1)
                return target
        raise ValueError(f"Target tensor should have 0, 1 or 2 dimensions, but got {target.dim()} dimensions")

    @singledispatchmethod
    def get_targets(self, model_inputs):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}"
        )

    @get_targets.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.get_logits(model_inputs).argmax(dim=-1)

    @get_targets.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]]) -> Iterator:
        yield from (prediction.argmax(dim=-1) for prediction in self.get_logits(iter(model_inputs)))

    @singledispatchmethod
    def get_targeted_logits(self, model_inputs, targets: torch.Tensor):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_target_logits in class {self.__class__.__name__}"
        )

    @get_targeted_logits.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: torch.Tensor):
        logits = self.get_logits(model_inputs)
        targets = self._process_target(targets, logits)
        return logits.gather(-1, targets)

    @get_targeted_logits.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor):
        predictions = self.get_logits(iter(model_inputs))
        # TODO : refaire ça proprement
        if targets.dim() in (0, 1):
            targets = targets.view(1, -1)
        single_index = int(targets.shape[0] > 1)
        for index, logits in enumerate_generator(predictions):
            yield logits.gather(-1, targets[single_index and index].unsqueeze(0).expand(logits.shape[0], -1))

    @singledispatchmethod
    def get_gradients(self, model_inputs, targets: torch.Tensor):
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register(Mapping)
    def _(self, model_inputs: Mapping[str, torch.Tensor], targets: torch.Tensor):
        model_inputs = self.embed(model_inputs)

        def f(embed):
            return self.get_targeted_logits(embed, targets)

        return (
            torch.autograd.functional.jacobian(f, model_inputs["inputs_embeds"], create_graph=True, strict=False)
            .sum(dim=1)
            .abs()
            .mean(axis=-1)
        )

    @get_gradients.register(Iterable)
    def _(self, model_inputs: Iterable[Mapping[str, torch.Tensor]], targets: torch.Tensor):
        # TODO : see if we can do that without using persistent generator
        mi = PersistentGenerator(iter(model_inputs))
        for index, element in enumerate_generator(self.get_targeted_logits(mi, targets)):
            element.grad = None
            element.backward(torch.ones_like(element), retain_graph=True)
            return mi[index]["inputs_embeds"].grad.abs().mean(axis=-1)
