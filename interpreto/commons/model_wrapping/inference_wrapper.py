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

import torch
from transformers import PreTrainedModel


class InferenceWrapper:
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
        valid_keys = ["input_ids", "inputs_embeds", "attention_mask"]
        inputs = {key: value.to(self.device) for key, value in model_inputs.items() if key in valid_keys}
        return self.model(**inputs)
