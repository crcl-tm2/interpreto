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
Tests for interpreto.concepts.methods.concept_bottleneck methods
"""

from __future__ import annotations

import pytest
import torch

from interpreto.commons.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.concepts import (
    BatchTopKSAEConcepts,
    Cockatiel,
    ConceptAutoEncoderExplainer,
    ConvexNMFConcepts,
    DictionaryLearningConcepts,
    ICAConcepts,
    JumpReLUSAEConcepts,
    KMeansConcepts,
    NeuronsAsConcepts,
    NMFConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SparsePCAConcepts,
    SVDConcepts,
    TopKSAEConcepts,
    VanillaSAEConcepts,
)
from interpreto.concepts.methods.overcomplete import DictionaryLearningExplainer, SAEExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_CONCEPT_METHODS = [
    BatchTopKSAEConcepts,
    Cockatiel,
    ConvexNMFConcepts,
    DictionaryLearningConcepts,
    ICAConcepts,
    JumpReLUSAEConcepts,
    KMeansConcepts,
    NeuronsAsConcepts,
    NMFConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SparsePCAConcepts,
    SVDConcepts,
    TopKSAEConcepts,
    VanillaSAEConcepts,
]


def test_cbe_fit_failure_cases(splitted_encoder_ml: ModelWithSplitPoints):
    """Test failure cases in matching the CBE with a ModelWithSplitPoints"""
    initial_split_points = splitted_encoder_ml.split_points
    splitted_encoder_ml.split_points = [
        "cls.predictions.transform.LayerNorm",
        "bert.encoder.layer.1",
        "bert.encoder.layer.3.attention.self.query",
    ]

    # Raise when no split is provided and the model has more than one split
    with pytest.raises(ValueError, match="If the model has more than one split point"):
        _ = Cockatiel(splitted_encoder_ml, nb_concepts=3)

    # Restore initial split points
    splitted_encoder_ml.split_points = initial_split_points  # type: ignore


@pytest.mark.parametrize("method_class", ALL_CONCEPT_METHODS)
def test_overcomplete_cbe(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
    method_class: type[ConceptAutoEncoderExplainer],
):
    """Test SAEExplainer and DictionaryLearningExplainer"""
    split, activations = next(iter(activations_dict.items()))  # dictionary with only one element
    n = activations.shape[0]
    d = activations.shape[1]
    nb_concepts = 3

    # iterate over all methods from the namedtuple listing them
    if method_class == NeuronsAsConcepts:
        cbe = method_class(splitted_encoder_ml)  # type: ignore
    elif method_class in [Cockatiel, NMFConcepts]:
        cbe = method_class(
            splitted_encoder_ml,
            nb_concepts=nb_concepts,  # type: ignore
            device=DEVICE,  # type: ignore
            force_relu=True,  # type: ignore
        )  # type: ignore
        cbe.fit(activations)
    elif issubclass(method_class, SAEExplainer):
        cbe = method_class(splitted_encoder_ml, nb_concepts=nb_concepts, device=DEVICE)
        cbe.fit(activations, nb_epochs=1, batch_size=1, device=DEVICE)
    elif issubclass(method_class, DictionaryLearningExplainer):
        cbe = method_class(
            splitted_encoder_ml,
            nb_concepts=nb_concepts,
            device=DEVICE,
        )
        cbe.fit(activations)
    else:
        raise ValueError(f"Unknown method_class {method_class}")

    assert hasattr(cbe, "concept_model")
    assert hasattr(cbe.concept_model, "nb_concepts")
    assert hasattr(cbe, "model_with_split_points")
    assert cbe.concept_model.fitted
    assert cbe.is_fitted
    assert cbe.split_point == split
    assert hasattr(cbe, "has_differentiable_concept_encoder")
    assert hasattr(cbe, "has_differentiable_concept_decoder")

    concepts = cbe.encode_activations(activations)
    reconstructed_activations = cbe.decode_concepts(concepts)
    assert reconstructed_activations.shape == (n, d)

    dictionary = cbe.get_dictionary()
    if method_class == NeuronsAsConcepts:
        assert cbe.concept_model.nb_concepts == d
        assert concepts.shape == (n, d)
        assert torch.allclose(dictionary, torch.eye(d))
    else:
        assert cbe.concept_model.nb_concepts == nb_concepts
        assert concepts.shape == (n, nb_concepts)
        assert dictionary.shape == (nb_concepts, d)
