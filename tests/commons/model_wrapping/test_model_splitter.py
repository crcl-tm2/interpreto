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

from interpreto.commons.model_wrapping.model_splitter import ModelSplitter


@fixture
def encoder_lm_splitter() -> ModelSplitter:
    return ModelSplitter(
        "huawei-noah/TinyBERT_General_4L_312D",
        splits=[],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )


BERT_SPLITS = [
    "cls.predictions.transform.LayerNorm",
    "bert.encoder.layer.1",
    "bert.encoder.layer.3.attention.self.query",
]

BERT_SPLITS_SORTED = [
    "bert.encoder.layer.1",
    "bert.encoder.layer.3.attention.self.query",
    "cls.predictions.transform.LayerNorm",
]


def test_order_splits(encoder_lm_splitter: ModelSplitter):
    """
    Test the _order_splits method with various scenarios
    """
    encoder_lm_splitter.splits = BERT_SPLITS
    # Assert the ordered splits match expected order
    assert encoder_lm_splitter.splits == BERT_SPLITS_SORTED, (
        f"Failed for splits: {BERT_SPLITS}\nExpected: {BERT_SPLITS_SORTED}\nGot:      {encoder_lm_splitter.splits}"
    )


def test_activation_equivalence_batched_text_token_inputs(encoder_lm_splitter: ModelSplitter):
    """
    Test the equivalence of activations for text and token inputs
    """
    encoder_lm_splitter.splits = BERT_SPLITS
    txt = ["Hello, my dog is cute", "The cat is on the [MASK]"]
    tok_inputs = encoder_lm_splitter.tokenizer(txt, return_tensors="pt")

    activations_txt = encoder_lm_splitter.get_activations(txt)
    activations_ids = encoder_lm_splitter.get_activations(tok_inputs)

    for k in activations_txt.keys():
        assert torch.allclose(activations_txt[k], activations_ids[k])


def test_index_by_layer_idx(encoder_lm_splitter: ModelSplitter):
    """Test indexing by layer idx"""
    splits_with_layer_idx: list[str | int] = list(BERT_SPLITS)
    splits_with_layer_idx[1] = 1  # instead of bert.encoder.layer.1
    encoder_lm_splitter.splits = splits_with_layer_idx
    assert encoder_lm_splitter.splits == BERT_SPLITS_SORTED, (
        f"Failed for splits: {BERT_SPLITS}\nExpected: {BERT_SPLITS_SORTED}\nGot:      {encoder_lm_splitter.splits}"
    )
