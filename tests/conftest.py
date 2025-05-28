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
Common fixtures for all tests
"""

from pytest import fixture
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

from interpreto.commons import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.typing import LatentActivations


@fixture(scope="session")
def sentences():
    return [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "Interpreto is magical",
        "Testing interpreto",
    ]


@fixture(scope="session")
def multi_split_model() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=[
            "cls.predictions.transform.LayerNorm",
            "bert.encoder.layer.1.output",
            "bert.encoder.layer.3.attention.self.query",
        ],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


@fixture(scope="session")
def splitted_encoder_ml() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=["bert.encoder.layer.1.output"],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


@fixture(scope="session")
def activations_dict(splitted_encoder_ml: ModelWithSplitPoints, sentences: list[str]) -> dict[str, LatentActivations]:
    return splitted_encoder_ml.get_activations(sentences, select_strategy=ActivationSelectionStrategy.FLATTEN)  # type: ignore


@fixture(scope="session")
def model():
    return AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


@fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
