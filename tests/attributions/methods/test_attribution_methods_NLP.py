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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    # AutoModelForMultipleChoice,
    # AutoModelForQuestionAnswering,
    # AutoModelForTokenClassification,
    AutoTokenizer,
)

from interpreto.attributions import (
    IntegratedGradients,
    KernelShap,
    Lime,
    OcclusionExplainer,
    Saliency,
    SmoothGrad,
    SobolAttribution,
)
from interpreto.attributions.base import AttributionOutput
from interpreto.commons.granularity import GranularityLevel
from interpreto.commons.model_wrapping.inference_wrapper import InferenceModes
from interpreto.typing import IncompatibilityError

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

attribution_method_kwargs = {
    # -----------------------
    # Gradient based methods:
    Saliency: {},
    IntegratedGradients: {"n_interpolations": 3, "baseline": 0.0},
    SmoothGrad: {"n_interpolations": 3, "noise_level": 0.1},
    # ---------------------------
    # Perturbation based methods:
    OcclusionExplainer: {"granularity_level": GranularityLevel.TOKEN, "inference_mode": InferenceModes.SOFTMAX},
    KernelShap: {
        "n_perturbations": 3,
        "inference_mode": InferenceModes.LOG_SOFTMAX,
        "granularity_level": GranularityLevel.TOKEN,  # TODO: change to ALL_TOKENS
    },
    Lime: {"n_perturbations": 3, "granularity_level": GranularityLevel.WORD},
    SobolAttribution: {"n_token_perturbations": 3},
}


model_loader_combinations = {
    "hf-internal-testing/tiny-random-albert": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-bart": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-DebertaV2Model": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-distilbert": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-ElectraModel": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-roberta": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-t5": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-xlm-roberta": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-gpt2": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-gpt_neo": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-gptj": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-CodeGenForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-FalconModel": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-LlamaForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-MistralForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-Starcoder2ForCausalLM": AutoModelForCausalLM,
}

# TODO: ADD BACK TEST IMMEDIATELY. It was remove just to see if it is the problem for the CI
# @pytest.mark.parametrize("model_name", model_loader_combinations.keys())
# @pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
# def test_attribution_methods_with_text(model_name, attribution_explainer):
#     """Tests all combinations of models and loaders with an attribution method"""
#     model_loader = model_loader_combinations[model_name]

#     model = model_loader.from_pretrained(model_name).to(DEVICE)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     assert model is not None, f"Model loading failed {model_name}"
#     assert tokenizer is not None, f"Tokenizer failed to load for {model_name}"

#     # To be changed according to the final form of the explainer:
#     explainer_kwargs = attribution_method_kwargs.get(attribution_explainer, {})
#     explainer = attribution_explainer(model, tokenizer=tokenizer, batch_size=3, device=DEVICE, **explainer_kwargs)

#     # we need to test both type of inputs: text, list_text, tokenized_text, tokenized_list_text:
#     text = "He is my best friend"
#     list_text = [
#         "Short",
#         "Medium sentence length",
#         "Much longer sentence length, because we need to test different length of sentences",
#         "Interpreto is magic",
#     ]
#     list_input_text_onlytext = [text, text, list_text, list_text]

#     list_input_text_onlytokenized = [
#         tokenizer(input_text_onlytext, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#         for input_text_onlytext in [text, list_text]
#     ]

#     list_input_text = list_input_text_onlytext + list_input_text_onlytokenized

#     # we need to test with and without targets:
#     if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
#         list_target = [
#             None,
#             "and I like him.",
#             None,
#             ["sentence", "for testing", "that is good practice", "try it."],
#             None,
#             None,
#         ]
#     else:
#         list_target = [
#             None,
#             1,
#             None,
#             torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
#             None,
#             [0, 0, 1, 0],
#         ]

#     for input_text, target in zip(list_input_text, list_target, strict=False):
#         # if we have a generative model, we need to give the max_length:
#         try:
#             if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
#                 attributions = explainer.explain(input_text, targets=target, generation_kwargs={"max_length": 10})
#             else:
#                 attributions = explainer.explain(input_text, targets=target)
#         except IncompatibilityError:
#             continue

#         # Checks:
#         assert isinstance(attributions, list), "The output of the attribution explainer must be a list"

#         if isinstance(input_text, str):
#             assert len(attributions) == 1, (
#                 "The number of elements in the list must correspond to the number of inputs."
#             )
#         if isinstance(input_text, list):
#             assert len(attributions) == len(input_text), (
#                 "The number of elements in the list must correspond to the number of inputs."
#             )
#         if isinstance(input_text, dict):
#             assert len(attributions) == input_text["input_ids"].shape[0], (
#                 "The number of elements in the list must correspond to the number of inputs."
#             )
#         assert all(isinstance(attribution, AttributionOutput) for attribution in attributions), (
#             "The elements of the list must be of type AttributionOutput."
#         )
#         assert all(
#             len(attribution.elements) == (attribution.attributions).shape[-1] for attribution in attributions
#         ), "In the AttributionOutput class, elements and attributions must have the same length."
