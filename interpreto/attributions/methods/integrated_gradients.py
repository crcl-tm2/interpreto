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
Integrated Gradients method
"""

from __future__ import annotations

from typing import Any

import torch

from interpreto.attributions.aggregations import MeanAggregator
from interpreto.attributions.base import GradientExplainer
from interpreto.attributions.perturbations import LinearInterpolationPerturbator
from interpreto.commons.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper


class IntegratedGradients(GradientExplainer):
    """
    Integrated Gradients method
    """

    def __init__(
        self,
        model: Any,
        batch_size: int,
        device: torch.device | None = None,
        n_interpolations: int = 10,
        baseline: torch.Tensor | float | None = None,
    ):
        super().__init__(
            perturbator=LinearInterpolationPerturbator(baseline=baseline, n_perturbations=n_interpolations),
            inference_wrapper=ClassificationInferenceWrapper(model, batch_size=batch_size, device=device),
            aggregator=MeanAggregator(),  # TODO: check if we need a trapezoidal mean
            device=device,
        )
