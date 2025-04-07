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
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpreto.commons.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper


@pytest.fixture
def sentences():
    return [
        "Once upon a time, in a village nestled between two mountains, there lived a curious child named Elio.",
        "Write a short story about a robot who learns how to paint emotions.",
        "Describe a world where water flows upward instead of down.",
        "What would a conversation sound like between a dragon and a scientist?",
        "Explain quantum physics to a five-year-old using a bedtime story.",
        "Generate a poem about loneliness that ends on a hopeful note.",
        "Imagine a dialogue between the moon and the ocean.",
        "Continue the sentence: 'She opened the door and saw…'",
        "Invent a new holiday and describe how people celebrate it.",
        "Create a futuristic news headline for the year 3025.",
    ]


@pytest.fixture
def sentence():
    return "Once upon a time, in a village nestled between two mountains, there lived a curious child named Elio."


generation_models = ["hf-internal-testing/tiny-random-LlamaForCausalLM", "hf-internal-testing/tiny-random-gpt2"]


def prepare_model_and_tokenizer(model_name: str):
    """
    Helper function to prepare the tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inference_wrapper = GenerationInferenceWrapper(model, batch_size=5)
    return tokenizer, model, inference_wrapper


@pytest.mark.parametrize("model_name", generation_models)
def test_generation_inference_wrapper_single_sentence(model_name, sentence):
    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    full_model_inputs, target = inference_wrapper.get_inputs_to_explain_and_targets(
        model_inputs, max_length=6, do_sample=False
    )
    target_length = target.shape[1]
    full_shape = full_model_inputs["input_ids"].shape[1]
    assert full_model_inputs["input_ids"][:target_length].equal(
        model_inputs["input_ids"]
    )  # check that the first part of the full input is equal to the original input
    assert full_model_inputs["input_ids"][target_length:].equal(
        target
    )  # check that the second part of the full input is equal to the target

    logits = inference_wrapper.get_logits(full_model_inputs)
    assert logits.shape == (1, full_shape, model.config.vocab_size)  # check that the logits shape is correct

    targeted_logits = inference_wrapper.get_targeted_logits(full_model_inputs, target)
    assert targeted_logits.shape == (1, target_length)  # check that the targeted logits shape is correct

    grad_matrix = inference_wrapper.get_gradients(full_model_inputs, target)
    assert grad_matrix.shape == (1, target_length, full_shape)  # check that the grad matrix shape is correct


@pytest.mark.parametrize("model_name", generation_models)
def test_generation_inference_wrapper_multiple_sentences(model_name, sentences):
    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_sentences = len(sentences)
    model_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    full_model_inputs, target = inference_wrapper.get_inputs_to_explain_and_targets(
        model_inputs, max_length=6, do_sample=False
    )
    target_length = target.shape[1]
    full_shape = full_model_inputs["input_ids"].shape[1]
    assert full_model_inputs["input_ids"][:target_length].equal(
        model_inputs["input_ids"]
    )  # check that the first part of the full input is equal to the original input
    assert full_model_inputs["input_ids"][target_length:].equal(
        target
    )  # check that the second part of the full input is equal to the target

    logits = inference_wrapper.get_logits(full_model_inputs)
    assert logits.shape == (n_sentences, full_shape, model.config.vocab_size)  # check that the logits shape is correct

    targeted_logits = inference_wrapper.get_targeted_logits(full_model_inputs, target)
    assert targeted_logits.shape == (n_sentences, target_length)  # check that the targeted logits shape is correct

    grad_matrix = inference_wrapper.get_gradients(full_model_inputs, target)
    assert grad_matrix.shape == (n_sentences, target_length, full_shape)  # check that the grad matrix shape is correct


@pytest.mark.parametrize("model_name", generation_models)
def test_generation_inference_wrapper_multiple_mappings(model_name, sentences):
    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_sentences = len(sentences)
    nb_split = 3
    if nb_split >= n_sentences:
        raise ValueError("nb_split must be less than the number of sentences (n_sentences).")

    model_inputs1 = tokenizer(sentences[:nb_split], return_tensors="pt", padding=True, truncation=True)
    model_inputs2 = tokenizer(sentences[nb_split:], return_tensors="pt", padding=True, truncation=True)
    model_inputs = [model_inputs1, model_inputs2]

    full_model_inputs, target = inference_wrapper.get_inputs_to_explain_and_targets(
        model_inputs, max_length=6, do_sample=False
    )
    target_length1 = target[0].shape[1]
    target_length2 = target[1].shape[1]
    full_shape1 = full_model_inputs[0]["input_ids"].shape[1]
    full_shape2 = full_model_inputs[1]["input_ids"].shape[1]

    # check that the first part of the full input is equal to the original input:
    assert full_model_inputs[0]["input_ids"][:target_length1].equal(model_inputs[0]["input_ids"])
    assert full_model_inputs[1]["input_ids"][:target_length2].equal(model_inputs[1]["input_ids"])
    # check that the second part of the full input is equal to the target:
    assert full_model_inputs[0]["input_ids"][target_length1:].equal(target[0])
    assert full_model_inputs[1]["input_ids"][target_length2:].equal(target[1])

    logits = inference_wrapper.get_logits(full_model_inputs)
    # check that the logits shape is correct:
    assert logits[0].shape == (nb_split, full_shape1, model.config.vocab_size)
    assert logits[1].shape == (n_sentences - nb_split, full_shape2, model.config.vocab_size)

    targeted_logits = inference_wrapper.get_targeted_logits(full_model_inputs, target)
    # check that the targeted logits shape is correct:
    assert targeted_logits[0].shape == (nb_split, target_length1)
    assert targeted_logits[1].shape == (n_sentences - nb_split, target_length2)

    grad_matrix = inference_wrapper.get_gradients(full_model_inputs, target)
    # check that the grad matrix shape is correct:
    assert grad_matrix[0].shape == (nb_split, target_length1, full_shape1)
    assert grad_matrix[1].shape == (n_sentences - nb_split, target_length2, full_shape2)
