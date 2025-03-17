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
Tests for `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

import pytest
import torch
from pytest import fixture
from transformers import AutoModelForMaskedLM

from interpreto.commons.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.concepts import NeuronsAsConcepts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@fixture
def encoder_lm_splitter() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=[],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


def test_overcomplete_cbe(encoder_lm_splitter: ModelWithSplitPoints):
    """
    Test that the concept encoding and decoding of the `NeuronsAsConcepts` is the identity
    """

    txt = ["Hello, my dog is cute", "The cat is on the [MASK]"]
    split = "bert.encoder.layer.1.output"
    encoder_lm_splitter.split_points = split
    activations = encoder_lm_splitter.get_activations(txt, select_strategy="flatten")
    assert activations[split].shape == (16, 312)

    concept_explainer = NeuronsAsConcepts(model_with_split_points=encoder_lm_splitter, split_point=split)

    assert concept_explainer.is_fitted is True  # splitted_model has a single split so it is fitted
    assert concept_explainer.split_point == split
    assert hasattr(concept_explainer, "has_differentiable_concept_encoder")
    assert hasattr(concept_explainer, "has_differentiable_concept_decoder")
    assert concept_explainer.concept_model.nb_concepts == 312

    concepts = concept_explainer.encode_activations(activations[split])
    reconstructed_activations = concept_explainer.decode_concepts(concepts)

    assert torch.allclose(concepts, activations[split])
    assert torch.allclose(reconstructed_activations, activations[split])

    with pytest.raises(NotImplementedError):
        concept_explainer.fit(activations[split])
