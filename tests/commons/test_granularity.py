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
from transformers import AutoTokenizer

from interpreto.commons.granularity import Granularity


class DummyMapping(dict):
    """Simple mapping implementing ``word_ids`` for testing."""

    def __init__(self):
        input_ids = torch.tensor(
            [
                [101, 5, 6, 7, 102],
                [101, 8, 9, 102, 0],
            ]
        )
        special_tokens_mask = torch.tensor(
            [
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 1],
            ]
        )
        super().__init__(input_ids=input_ids, special_tokens_mask=special_tokens_mask)
        self._word_ids = [
            [None, 0, 0, 1, None],
            [None, 0, 1, None, None],
        ]

    def word_ids(self, index: int):
        return self._word_ids[index]


tokens = DummyMapping()


def test_get_length():
    assert torch.equal(
        Granularity.get_length(tokens, Granularity.ALL_TOKENS),
        torch.tensor([5, 5]),
    )
    assert torch.equal(
        Granularity.get_length(tokens, Granularity.TOKEN),
        torch.tensor([3, 2]),
    )
    assert torch.equal(
        Granularity.get_length(tokens, Granularity.WORD),
        torch.tensor([2, 2]),
    )


def test_get_decomposition():
    assert Granularity.get_decomposition(tokens, Granularity.ALL_TOKENS) == [
        [[101], [5], [6], [7], [102]],
        [[101], [8], [9], [102], [0]],
    ]
    assert Granularity.get_decomposition(tokens, Granularity.TOKEN) == [
        [[5], [6], [7]],
        [[8], [9]],
    ]
    assert Granularity.get_decomposition(tokens, Granularity.WORD) == [
        [[5, 6], [7]],
        [[8], [9]],
    ]


def test_get_association_matrix():
    all_tokens = Granularity.get_association_matrix(tokens, Granularity.ALL_TOKENS)
    assert torch.equal(all_tokens[0], torch.eye(5))
    assert torch.equal(all_tokens[1], torch.eye(5))

    token_matrix = Granularity.get_association_matrix(tokens, Granularity.TOKEN)
    expected_token = torch.tensor(
        [
            [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
        ]
    )
    assert torch.equal(token_matrix, expected_token)

    word_matrix = Granularity.get_association_matrix(tokens, Granularity.WORD)
    expected_word = torch.tensor(
        [
            [[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
        ]
    )
    assert torch.equal(word_matrix, expected_word)


def test_granularity_on_text():
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "word longword verylongword"
    # bert_tokenizer.tokenize(text)
    # > ['word', 'long', '##word', 'very', '##long', '##word']
    tokens = bert_tokenizer(text, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True)
    all_tokens = Granularity.get_association_matrix(tokens, Granularity.ALL_TOKENS)
    assert torch.equal(all_tokens[0], torch.eye(8))

    token_matrix = Granularity.get_association_matrix(tokens, Granularity.TOKEN)
    expected_token = torch.tensor(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
        ]
    )
    assert torch.equal(token_matrix, expected_token)

    word_matrix = Granularity.get_association_matrix(tokens, Granularity.WORD)
    expected_word = torch.tensor(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
            ],
        ]
    )
    assert torch.equal(word_matrix, expected_word)


if __name__ == "__main__":
    test_get_length()
    test_get_decomposition()
    test_get_association_matrix()
    test_granularity_on_text()
