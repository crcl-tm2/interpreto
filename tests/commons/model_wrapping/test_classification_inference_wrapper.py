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

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interpreto.commons.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper


@pytest.fixture
def sentences():
    return [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    ]


@pytest.fixture
def sentence():
    return "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."


classification_models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
]


def prepare_model_and_tokenizer(model_name: str):
    """
    Helper function to prepare the tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inference_wrapper = ClassificationInferenceWrapper(model, batch_size=5)
    return tokenizer, model, inference_wrapper


@pytest.mark.parametrize("model_name", classification_models)
def test_classification_inference_wrapper_single_sentence(model_name, sentence):
    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)

    # Reference values
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    logits = model(**tokens).logits
    target = logits.argmax(dim=-1)
    predefined_targets = torch.randperm(logits.shape[-1])
    scores = logits.index_select(dim=-1, index=predefined_targets)

    # Tests
    assert torch.equal(logits, inference_wrapper.get_logits(tokens))
    assert torch.equal(logits, next(inference_wrapper.get_logits([tokens])))
    assert torch.equal(target, inference_wrapper.get_targets(tokens))
    assert torch.equal(target, next(inference_wrapper.get_targets([tokens])))
    assert torch.equal(scores, inference_wrapper.get_target_logits(tokens, predefined_targets))
    assert torch.equal(scores, next(inference_wrapper.get_target_logits([tokens], predefined_targets)))


@pytest.mark.parametrize("model_name", classification_models)
def test_classification_inference_wrapper_multiple_sentences(model_name, sentences):
    ### Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)

    ### Reference values
    tokens = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    logits = model(**tokens).logits
    targets = logits.argmax(dim=-1)
    predefined_targets = torch.randperm(logits.shape[-1])
    target_logits = torch.gather(logits, dim=-1, index=predefined_targets.unsqueeze(0).expand(logits.shape[0], -1))

    ### Tests
    # TODO : check why they are not equal
    assert torch.equal(logits, torch.stack(list(inference_wrapper.get_logits(tokens))))
    assert torch.equal(targets, torch.stack(list(inference_wrapper.get_targets(tokens))))
    assert torch.equal(
        target_logits, torch.stack(list(inference_wrapper.get_target_logits(tokens, predefined_targets)))
    )
