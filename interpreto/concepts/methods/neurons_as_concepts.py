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
Concept Bottleneck Explainer based on Overcomplete concept-encoder-decoder framework.
"""

from __future__ import annotations

from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.base import ConceptBottleneckExplainer
from interpreto.typing import ConceptsActivations, LatentActivations


class NeuronsAsConcepts(ConceptBottleneckExplainer):
    """
    Concept Bottleneck Explainer where the latent space is concidered as the concept space.
    """

    _differentiable_concept_decoder: bool = True
    _differentiable_concept_encoder: bool = True

    def __init__(
        self,
        splitted_model: ModelSplitterPlaceholder,
    ):
        """
        Initialize the concept bottleneck explainer with the splitted model.

        Args:
            splitted_model (ModelSplitterPlaceholder):The model to apply the explanation on. It should be splitted between at least two parts.
        """
        super().__init__(splitted_model)
        if len(self.splitted_model.splits) == 1:
            self.split: str = self.splitted_model.splits[0]
        else:
            self.split: str | None = None

    @property
    def fitted(self) -> bool:
        return hasattr(self, "split") and self.split is not None

    def fit(self, activations: dict[str, LatentActivations] | None, split: str | None = None):
        _, split = self.verify_activations(activations)
        self.split: str = split

    def encode_activations(
        self, activations: LatentActivations | dict[str, LatentActivations], **kwargs
    ) -> ConceptsActivations:
        """
        Return the provided activations as concepts.

        Args:
            activations (LatentActivations | dict[str, LatentActivations]): The activations to encode.

        Returns:
            ConceptsActivations: The input activations on the model split.
        """
        if isinstance(activations, dict):
            inputs, _ = self.verify_activations(activations)
            return inputs

        return activations

    def decode_concepts(self, concepts: ConceptsActivations, **kwargs) -> LatentActivations:
        """
        Return the provided concepts as activations.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            LatentActivations: The input concepts returned as the reconstructed activations.
        """
        return concepts
