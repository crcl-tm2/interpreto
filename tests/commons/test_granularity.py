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
        super().__init__(input_ids=input_ids)
        self._word_ids = [
            [None, 0, 0, 1, None],
            [None, 0, 1, None, None],
        ]

    def word_ids(self, index: int):
        return self._word_ids[index]


tokens = DummyMapping()


def test_get_decomposition(real_bert_tokenizer):
    assert Granularity.get_decomposition(tokens, Granularity.ALL_TOKENS) == [
        [[101], [5], [6], [7], [102]],
        [[101], [8], [9], [102], [0]],
    ]
    assert Granularity.get_decomposition(tokens, Granularity.TOKEN, real_bert_tokenizer) == [
        [[5], [6], [7]],
        [[8], [9]],
    ]
    assert Granularity.get_decomposition(tokens, Granularity.WORD, real_bert_tokenizer) == [
        [[5, 6], [7]],
        [[8], [9]],
    ]


def test_get_association_matrix(real_bert_tokenizer):
    all_tokens = Granularity.get_association_matrix(tokens, Granularity.ALL_TOKENS)
    assert torch.equal(all_tokens[0], torch.eye(5))
    assert torch.equal(all_tokens[1], torch.eye(5))

    token_matrix = Granularity.get_association_matrix(tokens, Granularity.TOKEN, real_bert_tokenizer)
    expected_token = [
        torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=torch.bool),
        torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.bool),
    ]
    assert torch.equal(token_matrix[0], expected_token[0])
    assert torch.equal(token_matrix[1], expected_token[1])

    word_matrix = Granularity.get_association_matrix(tokens, Granularity.WORD, real_bert_tokenizer)
    expected_word = [
        torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=torch.bool),
        torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.bool),
    ]
    assert torch.equal(word_matrix[0], expected_word[0])
    assert torch.equal(word_matrix[1], expected_word[1])


def test_granularity_on_text(real_bert_tokenizer):
    text = "word longword verylongword"
    # real_bert_tokenizer.tokenize(text)
    # > ['word', 'long', '##word', 'very', '##long', '##word']
    tokens = real_bert_tokenizer(text, return_tensors="pt")
    all_tokens = Granularity.get_association_matrix(tokens, Granularity.ALL_TOKENS)
    assert torch.equal(all_tokens[0], torch.eye(8))

    token_matrix = Granularity.get_association_matrix(tokens, Granularity.TOKEN, real_bert_tokenizer)
    assert len(token_matrix) == 1
    token_matrix = token_matrix[0]
    expected_token = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(token_matrix, expected_token)

    word_matrix = Granularity.get_association_matrix(tokens, Granularity.WORD, real_bert_tokenizer)
    assert len(word_matrix) == 1
    word_matrix = word_matrix[0]
    expected_word = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
        ],
    )
    assert torch.equal(word_matrix, expected_word)


def test_granularity_on_text_padding(real_bert_tokenizer):
    # First sentence has 2 words → 3 sub-tokens (+2 specials) = 5,
    # second has 3 words → 6 sub-tokens (+2 specials) = 8.
    # With padding='longest' the first sequence is padded to length 8.
    texts = [
        "word longword",  # needs padding
        "word longword verylongword",  # the longest one
    ]

    tokens = real_bert_tokenizer(
        texts,
        padding=True,  # force [PAD] on the shorter sequence
        return_tensors="pt",
    )

    # ------------- ALL_TOKENS -----------------
    all_tokens = Granularity.get_association_matrix(tokens, Granularity.ALL_TOKENS)
    # Both sequences are length-8 and should be pure identities (incl. PAD slots)
    assert torch.equal(all_tokens[0], torch.eye(8))
    assert torch.equal(all_tokens[1], torch.eye(8))

    # ------------- TOKEN ----------------------
    token_matrix = Granularity.get_association_matrix(tokens, Granularity.TOKEN, real_bert_tokenizer)
    # max_token_count = 6 (from the longer sentence)
    expected_token = [
        # sample 0 (3 real tokens, then zero-rows)
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],  # 'word'
                [0, 0, 1, 0, 0, 0, 0, 0],  # 'long'
                [0, 0, 0, 1, 0, 0, 0, 0],  # '##word'
            ],
            dtype=torch.bool,
        ),
        # sample 1 (6 real tokens)
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],  # 'word'
                [0, 0, 1, 0, 0, 0, 0, 0],  # 'long'
                [0, 0, 0, 1, 0, 0, 0, 0],  # '##word'
                [0, 0, 0, 0, 1, 0, 0, 0],  # 'very'
                [0, 0, 0, 0, 0, 1, 0, 0],  # '##long'
                [0, 0, 0, 0, 0, 0, 1, 0],  # '##word'
            ],
            dtype=torch.bool,
        ),
    ]
    assert torch.equal(token_matrix[0], expected_token[0])
    assert torch.equal(token_matrix[1], expected_token[1])

    # ------------- WORD -----------------------
    word_matrix = Granularity.get_association_matrix(tokens, Granularity.WORD, real_bert_tokenizer)
    # max_word_count = 3 (from the longer sentence)
    expected_word = [
        # sample 0 (2 real words, then zero-row)
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],  # 'word'
                [0, 0, 1, 1, 0, 0, 0, 0],  # 'longword'
            ],
            dtype=torch.bool,
        ),
        # sample 1 (3 real words)
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],  # 'word'
                [0, 0, 1, 1, 0, 0, 0, 0],  # 'longword'
                [0, 0, 0, 0, 1, 1, 1, 0],  # 'verylongword'
            ],
            dtype=torch.bool,
        ),
    ]
    assert torch.equal(word_matrix[0], expected_word[0])
    assert torch.equal(word_matrix[1], expected_word[1])


if __name__ == "__main__":
    real_bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_get_decomposition(real_bert_tokenizer)
    test_get_association_matrix(real_bert_tokenizer)
    test_granularity_on_text(real_bert_tokenizer)
    test_granularity_on_text_padding(real_bert_tokenizer)
