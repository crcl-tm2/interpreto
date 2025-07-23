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
"""Perturbation for GradientSHAP."""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from interpreto.attributions.perturbations.base import Perturbator
from interpreto.attributions.perturbations.linear_interpolation_perturbation import (
    LinearInterpolationPerturbator,
)
from interpreto.typing import TensorBaseline, TensorMapping


class GradientShapPerturbator(Perturbator):
    """Generate samples for GradientSHAP."""

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        baseline: TensorBaseline = None,
        n_perturbations: int = 10,
        *,
        std: float = 0.1,
    ) -> None:
        super().__init__(inputs_embedder)
        self.baseline = baseline
        self.n_perturbations = n_perturbations
        self.std = std

    @jaxtyped(typechecker=beartype)
    def perturb_embeds(self, model_inputs: TensorMapping) -> tuple[TensorMapping, None]:
        embeddings = model_inputs["inputs_embeds"]  # (b, l, d)
        baseline = LinearInterpolationPerturbator.adjust_baseline(self.baseline, embeddings)
        baseline = baseline.to(embeddings.device)
        b = embeddings.shape[0]
        baseline = baseline.unsqueeze(0).expand(b, *baseline.shape)

        baseline = baseline.unsqueeze(0).repeat(self.n_perturbations, 1, 1, 1)
        baseline += torch.randn_like(baseline) * self.std

        embeddings = embeddings.unsqueeze(0).repeat(self.n_perturbations, 1, 1, 1)
        alphas = torch.rand(self.n_perturbations, 1, 1, 1, device=embeddings.device)

        model_inputs["inputs_embeds"] = (1 - alphas) * baseline + alphas * embeddings
        model_inputs["inputs_embeds"] = model_inputs["inputs_embeds"].view(
            self.n_perturbations * b, *embeddings.shape[2:]
        )
        model_inputs["attention_mask"] = (
            model_inputs["attention_mask"]
            .unsqueeze(0)
            .repeat(self.n_perturbations, 1, 1)
            .reshape(self.n_perturbations * b, -1)
        )
        return model_inputs, None
