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
Base classes for perturbations used in attribution methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable

import torch

from interpreto.typing import ModelInput


class Perturbator(ABC):
    """
    Object allowing you to perturb an input (add noise, change tokens, create progression of vectors...)
    """

    @abstractmethod
    def perturb(
        self, inputs: ModelInput | Collection[ModelInput]
    ) -> tuple[torch.Tensor] | tuple[Iterable[torch.Tensor]]:
        """
        Method to perturb an input, should return a collection of perturbed elements and their associated masks
        """


class TokenPerturbator(Perturbator):
    """
    Generic class for token modification (occlusion, words substitution...)
    """


class TensorPerturbator(Perturbator):
    """
    Generic class for any tensor-wise modification
    """


class GaussianNoisePerturbator(TensorPerturbator):
    """
    Perturbator adding gaussian noise to the input tensor
    """

    def __init__(self, n_perturbations: int = 10, *, std: float = 0.1):
        self.n_perturbations = n_perturbations
        self.std = std

    def perturb(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        noise = torch.randn(self.n_perturbations, *inputs.shape, device=inputs.device) * self.std
        return inputs + noise, None
