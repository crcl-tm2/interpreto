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

from typing import Any

import torch
from nnsight.intervention.graph import InterventionProxy

from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.typing import ConceptsActivations, LatentActivations


class ConSim:
    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
        judge_llm: Any,
    ):
        self.concept_explainer = concept_explainer

    def extract_interesting_samples(
        self,
        inputs: list[str],
        labels: list[str],
        predictions: list[str],
        seed: int = 0,
    ) -> list[int]:
        raise NotImplementedError
        interesting_indices: list[int]
        return interesting_indices

    def compute_elements(
        self,
        interesting_samples: list[str] | None = None,
        interesting_indices: list[int] | None = None,
    ) -> tuple[LatentActivations, ConceptsActivations, torch.Tensor]:
        raise NotImplementedError
        interesting_latent_activations: LatentActivations
        interesting_concepts_activations: ConceptsActivations
        concepts_importance: torch.tensor
        return latent_activations, concepts_activations, concepts_importance

    def generate_prompt(
        self,
        interesting_samples: list[str],
        latent_activations: LatentActivations | InterventionProxy,
        concepts_activations: ConceptsActivations,
        labels: list[int],
        predictions: list[int],
    ) -> dict[str, str]:
        raise NotImplementedError
        prompt: dict[str, str]
        return prompt
