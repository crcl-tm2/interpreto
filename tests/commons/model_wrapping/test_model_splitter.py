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

from transformers import AutoModelForMaskedLM

from interpreto.commons.model_wrapping.model_splitter import ModelSplitter


def test_order_splits():
    """
    Test the _order_splits method with various scenarios
    """
    test_cases = [
        # Completely out of order splits
        {
            "splits": [
                "model.bert.embeddings",
                "model.bert.encoder.layer.3.attention.self.dropout",
                "model.bert.encoder.layer.1",
                "model.bert.encoder.layer.1.attention.output.dense",
            ],
            "expected_order": [
                "model.bert.embeddings",
                "model.bert.encoder.layer.1.attention.output.dense",
                "model.bert.encoder.layer.1",
                "model.bert.encoder.layer.3.attention.self.dropout",
            ],
        },
        # Already ordered splits
        {
            "splits": [
                "model.bert.embeddings",
                "model.bert.encoder.layer.1.attention.output.dense",
                "model.bert.encoder.layer.1",
                "model.bert.encoder.layer.3.attention.self.dropout",
            ],
            "expected_order": [
                "model.bert.embeddings",
                "model.bert.encoder.layer.1.attention.output.dense",
                "model.bert.encoder.layer.1",
                "model.bert.encoder.layer.3.attention.self.dropout",
            ],
        },
    ]
    for case in test_cases:
        model: ModelSplitter = ModelSplitter(
            "huawei-noah/TinyBERT_General_4L_312D",
            splits=case["splits"],
            model_class=AutoModelForMaskedLM,  # type: ignore
        )
        # Assert the ordered splits match expected order
        assert model.splits == case["expected_order"], (
            f"Failed for splits: {case['splits']}\nExpected: {case['expected_order']}\nGot:      {model.splits}"
        )
