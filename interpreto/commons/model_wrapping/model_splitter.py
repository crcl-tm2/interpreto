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

"""
Model splitting classes based on NNsight
"""

from __future__ import annotations

from collections import OrderedDict

from torch import nn

from interpreto.typing import LatentActivations, ModelInput


class ModelSplitterPlaceholder:
    """
    Placeholder class for model splitting
    """

    def __init__(self, model: nn.Module, splits: str | list[str]):
        assert splits == "input_to_latent"
        if isinstance(splits, str):
            splits = [splits]
        self.splits = splits
        assert hasattr(model, "input_to_latent") and hasattr(model, "end_model")
        self.model = model
        self.model_parts = OrderedDict(
            {
                "input_to_latent": self.model.input_to_latent,
                "end_model": self.model.end_model,
            }
        )
        self.latent_shape = self.model.fc3.in_features

    def get_activations(self, inputs: ModelInput) -> dict[str, LatentActivations]:
        """
        Get activations for a given input at each layer specified by the split
        """
        return {split: self.model_parts[split](inputs) for split in self.splits}
