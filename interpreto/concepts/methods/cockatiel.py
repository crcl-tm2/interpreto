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
Implementation of the Cockatiel concept explainer
"""

from __future__ import annotations

# from interpreto.attributions import Occlusion, SobolAttribution
from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.methods.overcomplete_cbe import OvercompleteDictionaryLearning, OvercompleteMethods
from interpreto.typing import ConceptsActivations, ModelInput


class Cockatiel(OvercompleteDictionaryLearning):
    """
    Implementation of the Cockatiel concept explainer

    Jourdan et al. - ACL 2023 - COCKATIEL: COntinuous Concept ranKed ATtribution with Interpretable ELements for explaining neural net classifiers on NLP
    https://aclanthology.org/2023.findings-acl.317/

    Attributes:
        splitted_model (ModelSplitterPlaceholder): Model splitter
        concept_encoder_decoder (oc_opt.NMF): Overcomplete NMF concept encoder decoder
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        _differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    def __init__(self, splitted_model: ModelSplitterPlaceholder, n_concepts: int, device: str = "cpu"):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete concept-encoder-decoder framework.

        Args:
            splitted_model (ModelSplitterPlaceholder): The model to apply the explanation on. It should be splitted between at least two parts.
            n_concepts (int): Number of concepts to explain.
            device (str): Device to use for the concept encoder-decoder.
        """
        super().__init__(
            splitted_model=splitted_model,
            ConceptEncoderDecoder=OvercompleteMethods.NMF,
            n_concepts=n_concepts,
            device=device,
        )

    def input_concept_attribution(
        self, inputs: ModelInput, concept: int | list[int], **attribution_kwargs
    ) -> list[float]:
        """
        Computes the attribution of each input to a given concept.

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences, or a dataset.
            concept (int | list[int]): The concept index (or list of concepts indices) to analyze.

        Returns:
            A list of attribution scores for each input.
        """
        return super().input_concept_attribution(
            inputs, concept, "Occlusion", **attribution_kwargs
        )  # TODO: add occlusion class when it exists

    def concept_output_attribution(self, concepts: ConceptsActivations, **attribution_kwargs):
        """
        Computes the attribution of each concept to a given example.

        Args:
            concepts (ConceptsActivations): The concepts to analyze.

        Returns:
            A list of attribution scores for each concept.
        """
        super().concept_output_attribution(
            concepts, attribution_method="SobolAttribution", **attribution_kwargs
        )  # TODO: add sobol class when it exists
