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
Base class for concept interpretation methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from interpreto.commons import ModelWithSplitPoints
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations, ModelInput


class BaseConceptInterpretationMethod(ABC):
    """Code: [:octicons-mark-github-24: `concepts/interpretations/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/interpretations/base.py)

    Abstract class defining an interface for concept interpretation.
    Its goal is to make the dimensions of the concept space interpretable by humans.

    Attributes:
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        split_point: str | None = None,
    ):
        if not hasattr(concept_model, "encode"):
            raise TypeError(
                f"Concept model should be able to encode activations into concepts. Got: {type(concept_model)}."
            )

        if split_point is None:
            if len(model_with_split_points.split_points) > 1:
                raise ValueError(
                    "If the model has more than one split point, a split point for fitting the concept model should "
                    f"be specified. Got split point: '{split_point}' with model split points: "
                    f"{', '.join(model_with_split_points.split_points)}."
                )
            split_point = model_with_split_points.split_points[0]

        if split_point not in model_with_split_points.split_points:
            raise ValueError(
                f"Split point '{split_point}' not found in model split points: "
                f"{', '.join(model_with_split_points.split_points)}."
            )

        self.model_with_split_points: ModelWithSplitPoints = model_with_split_points
        self.split_point: str = split_point
        self.concept_model: ConceptModelProtocol = concept_model

    @abstractmethod
    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: ModelInput | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Interpret the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and an object allowing to interpret them.
        It can be a label, a description, examples, etc.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (ModelInput | None): The inputs to use for the interpretation.
                Necessary if the source is not `VOCABULARY`, as examples are extracted from the inputs.
            latent_activations (LatentActivations | None): The latent activations to use for the interpretation.
                Necessary if the source is `LATENT_ACTIVATIONS`.
                Otherwise, it is computed from the inputs or ignored if the source is `CONCEPT_ACTIVATIONS`.
            concepts_activations (ConceptsActivations | None): The concepts activations to use for the interpretation.
                Necessary if the source is not `CONCEPT_ACTIVATIONS`. Otherwise, it is computed from the latent activations.

        Returns:
            Mapping[int, Any]: The interpretation of each of the specified concepts.
        """
        raise NotImplementedError
