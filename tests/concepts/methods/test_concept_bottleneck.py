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
from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae
from pytest import fixture
from transformers import AutoModelForMaskedLM

from interpreto.commons.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.concepts import (
    # Cockatiel,
    NeuronsAsConcepts,
    OvercompleteDictionaryLearning,
    OvercompleteOptimClasses,
    OvercompleteSAE,
    OvercompleteSAEClasses,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_CONCEPT_METHODS = list(OvercompleteSAEClasses) + list(OvercompleteOptimClasses) + [NeuronsAsConcepts]


@fixture
def encoder_lm_splitter() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=[],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


def test_cbe_fit_failure_cases(encoder_lm_splitter: ModelWithSplitPoints):
    """Test failure cases in matching the CBE with a ModelWithSplitPoints"""
    encoder_lm_splitter.split_points = [
        "cls.predictions.transform.LayerNorm",
        "bert.encoder.layer.1",
        "bert.encoder.layer.3.attention.self.query",
    ]

    # Raise when no split is provided and the model has more than one split
    with pytest.raises(ValueError, match="If the model has more than one split point"):
        cbe = OvercompleteDictionaryLearning(encoder_lm_splitter, oc_opt.NMF, nb_concepts=2)
        assert not cbe.is_fitted


@pytest.mark.slow
def test_overcomplete_cbe(encoder_lm_splitter: ModelWithSplitPoints):
    """Test OvercompleteSAE and OvercompleteDictionaryLearning"""

    latent_size = 312
    txt = ["Hello, my dog is cute", "The cat is on the [MASK]"]
    split = "bert.encoder.layer.1.output"
    nb_concepts = 3
    encoder_lm_splitter.split_points = split
    activations = encoder_lm_splitter.get_activations(txt, select_strategy="flatten")
    assert activations[split].shape == (16, latent_size)

    # iterate over all methods from the namedtuple listing them
    for method in ALL_CONCEPT_METHODS:
        if method == NeuronsAsConcepts:
            cbe = method(encoder_lm_splitter, split_point=split)
        elif issubclass(method.value, oc_sae.SAE):
            cbe = OvercompleteSAE(encoder_lm_splitter, method.value, nb_concepts=nb_concepts, device=DEVICE)
            cbe.fit(activations, nb_epochs=1, batch_size=1, device=DEVICE)
        elif issubclass(method.value, oc_opt.BaseOptimDictionaryLearning):
            cbe = OvercompleteDictionaryLearning(
                encoder_lm_splitter,
                method.value,
                nb_concepts=nb_concepts,
                device=DEVICE,
            )
            cbe.fit(activations)
        else:
            raise ValueError(f"Unknown method {method}")
        try:
            assert hasattr(cbe, "concept_model")
            assert hasattr(cbe.concept_model, "nb_concepts")
            assert hasattr(cbe, "model_with_split_points")
            assert cbe.concept_model.fitted
            assert cbe.is_fitted
            assert cbe.split_point == split
            assert hasattr(cbe, "has_differentiable_concept_encoder")
            assert hasattr(cbe, "has_differentiable_concept_decoder")

            concepts = cbe.encode_activations(activations[cbe.split_point])
            reconstructed_activations = cbe.decode_concepts(concepts)
            assert reconstructed_activations.shape == (16, latent_size)

            dictionary = cbe.get_dictionary()
            if method == NeuronsAsConcepts:
                assert cbe.concept_model.nb_concepts == latent_size
                assert concepts.shape == (16, latent_size)
                assert torch.allclose(dictionary, torch.eye(latent_size))
            else:
                assert cbe.concept_model.nb_concepts == nb_concepts
                assert concepts.shape == (16, nb_concepts)
                assert dictionary.shape == (nb_concepts, latent_size)
        except Exception as e:
            raise AssertionError(f"Error with {method}") from e
