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

from enum import Enum

from nnsight.intervention.graph import InterventionProxy

from interpreto.commons.distances import DistanceFunctionProtocol, DistanceFunctions
from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.typing import ConceptsActivations, LatentActivations


class ReconstructionSpaces(Enum):
    # TODO: docstring
    LATENT_ACTIVATIONS = "latent_activations"
    LOGITS = "logits"


class ReconstructionError:
    """Code [:octicons-mark-github-24: `concepts/metrics/faithfulness/reconstruction_error.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/faithfulness/reconstruction_error.py)

    TODO: docstring

    Attributes:
        concept_explainer (ConceptAutoEncoderExplainer): The explainer used to compute concepts.
        reconstruction_space (ReconstructionSpaces): The space in which the reconstruction error is computed.
        distance_function (DistanceFunctionProtocol): The distance function used to compute the reconstruction error.
    """

    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
        reconstruction_space: ReconstructionSpaces,
        distance_function: DistanceFunctionProtocol,
    ):
        self.concept_explainer = concept_explainer
        self.reconstruction_space = reconstruction_space
        self.distance_function = distance_function

    def compute(self, latent_activations: LatentActivations | InterventionProxy) -> float:
        """Compute the reconstruction error.

        Args:
            latent_activations (LatentActivations | InterventionProxy): The latent activations to use for the computation.

        Returns:
            float: The reconstruction error.
        """
        split_latent_activations: LatentActivations = self.concept_explainer._sanitize_activations(latent_activations)

        concepts_activations: ConceptsActivations = self.concept_explainer.encode_activations(split_latent_activations)

        reconstructed_latent_activations: LatentActivations = self.concept_explainer.decode_concepts(
            concepts_activations
        )

        if self.reconstruction_space is ReconstructionSpaces.LATENT_ACTIVATIONS:
            return self.distance_function(split_latent_activations, reconstructed_latent_activations).item()

        raise NotImplementedError("Only LATENT_ACTIVATIONS reconstruction space is supported.")


class LatentActivationsReconstructionError(ReconstructionError):
    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            reconstruction_space=ReconstructionSpaces.LATENT_ACTIVATIONS,
            distance_function=DistanceFunctions.EUCLIDEAN,
        )


class FID(ReconstructionError):
    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            reconstruction_space=ReconstructionSpaces.LATENT_ACTIVATIONS,
            distance_function=DistanceFunctions.WASSERSTEIN_1D,
        )


class Completeness(ReconstructionError):
    pass
