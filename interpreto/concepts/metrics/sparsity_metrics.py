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

import torch
from nnsight.intervention.graph import InterventionProxy

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.typing import ConceptsActivations, LatentActivations


class Sparsity:
    """Code [:octicons-mark-github-24: `concepts/metrics/complexity/sparsity.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/complexity/sparsity.py)

    TODO: docstring
    """

    def __init__(self, concept_explainer: ConceptEncoderExplainer, epsilon: float = 0.0):
        self.concept_explainer = concept_explainer
        self.epsilon = epsilon

    def compute(self, latent_activations: LatentActivations | InterventionProxy) -> float:
        """Compute the metric.

        Args:
            latent_activations (LatentActivations | InterventionProxy): The latent activations.

        Returns:
            float: The metric.
        """
        split_latent_activations: LatentActivations = self.concept_explainer._sanitize_activations(latent_activations)

        concepts_activations: ConceptsActivations = self.concept_explainer.encode_activations(split_latent_activations)

        return torch.mean(torch.abs(concepts_activations) > self.epsilon, dtype=torch.float32).item()


class SparsityRatio(Sparsity):
    """Code [:octicons-mark-github-24: `concepts/metrics/complexity/sparsity.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/complexity/sparsity.py)

    TODO: docstring
    """

    def compute(self, latent_activations: LatentActivations | InterventionProxy) -> float:
        """Compute the metric.

        Args:
            latent_activations (LatentActivations | InterventionProxy): The latent activations.

        Returns:
            float: The metric.
        """
        sparsity = super().compute(latent_activations)
        return sparsity / self.concept_explainer.concept_model.nb_concepts


# TODO: add hoyer and co, see overcomplete
