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

import torch
from pytest import fixture
from transformers import AutoModelForMaskedLM

from interpreto.commons.model_wrapping.model_with_split_points import ModelWithSplitPoints


@fixture
def encoder_lm() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "huawei-noah/TinyBERT_General_4L_312D",
        split_points=[],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


BERT_SPLIT_POINTS = [
    "cls.predictions.transform.LayerNorm",
    "bert.encoder.layer.1",
    "bert.encoder.layer.3.attention.self.query",
]

BERT_SPLIT_POINTS_SORTED = [
    "bert.encoder.layer.1",
    "bert.encoder.layer.3.attention.self.query",
    "cls.predictions.transform.LayerNorm",
]


def test_order_split_points(encoder_lm: ModelWithSplitPoints):
    """
    Test the sort_paths method upon split assignment
    """
    encoder_lm.split_points = BERT_SPLIT_POINTS
    # Assert the ordered split points match expected order
    assert encoder_lm.split_points == BERT_SPLIT_POINTS_SORTED, (
        f"Failed for split points: {BERT_SPLIT_POINTS}\n"
        f"Expected: {BERT_SPLIT_POINTS_SORTED}\n"
        f"Got:      {encoder_lm.split_points}"
    )


def test_activation_equivalence_batched_text_token_inputs(encoder_lm: ModelWithSplitPoints):
    """
    Test the equivalence of activations for text and token inputs
    """
    encoder_lm.split_points = BERT_SPLIT_POINTS
    txt = ["Hello, my dog is cute", "The cat is on the [MASK]"]
    tok_inputs = encoder_lm.tokenizer(txt, return_tensors="pt")

    activations_txt = encoder_lm.get_activations(txt)
    activations_ids = encoder_lm.get_activations(tok_inputs)

    for k in activations_txt.keys():
        assert torch.allclose(activations_txt[k], activations_ids[k])


def test_index_by_layer_idx(encoder_lm: ModelWithSplitPoints):
    """Test indexing by layer idx"""
    split_points_with_layer_idx: list[str | int] = list(BERT_SPLIT_POINTS)
    split_points_with_layer_idx[1] = 1  # instead of bert.encoder.layer.1
    encoder_lm.split_points = split_points_with_layer_idx
    assert encoder_lm.split_points == BERT_SPLIT_POINTS_SORTED, (
        f"Failed for split_points: {BERT_SPLIT_POINTS}\n"
        f"Expected: {BERT_SPLIT_POINTS_SORTED}\n"
        f"Got:      {encoder_lm.split_points}"
    )
