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
from __future__ import annotations

import torch
from nnsight import LanguageModel, NNsight
from pytest import fixture
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM, BertTokenizer


@fixture
def hf_lm_encoder_and_tokenizer() -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D", attn_implementation="eager"
    )
    tokenizer: BertTokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model.zero_grad()
    model.eval()
    return model, tokenizer


@fixture
def nnsight_lm_encoder() -> LanguageModel:
    model: LanguageModel = LanguageModel(
        "huawei-noah/TinyBERT_General_4L_312D",
        automodel=AutoModelForMaskedLM,
        tokenizer_kwargs={"padding_side": "right"},
        attn_implementation="eager",
    )
    return model


def test_attn_output_match_nnsight_hf_encoder_lm(hf_lm_encoder_and_tokenizer: AutoModelForMaskedLM):
    """Tests whether the extracted output attentions from NNsight match the output attentions from Huggingface.

    Args:
        hf_lm_encoder_and_tokenizer (AutoModelForMaskedLM): Huggingface model and tokenizer
    """
    hf_model, tokenizer = hf_lm_encoder_and_tokenizer
    nnsight_model = NNsight(hf_model)
    num_layers = len(nnsight_model.bert.encoder.layer)

    txt = "Hello, my dog is cute"
    inputs = tokenizer(txt, return_tensors="pt")
    inputs_tuple = (inputs["input_ids"], inputs["attention_mask"])

    attn_dic = {}

    with nnsight_model.trace(*inputs_tuple, invoker_kwargs={"output_attentions": True}):
        for i in range(num_layers):
            attn_dic[i] = nnsight_model.bert.encoder.layer[i].attention.self.output[1].save()
        nnsight_model.bert.encoder.layer[num_layers - 1].attention.self.output.stop()

    print("Attn dict keys:", attn_dic.keys())
    print("One attn entry shape:", attn_dic[0].shape)

    # Extract attention weights without nnsight
    out = hf_model(**inputs, output_attentions=True)

    # Check that weights in the two cases are identical
    for i in range(num_layers):
        assert torch.allclose(out.attentions[i], attn_dic[i])


def test_batched_nnsight_lm_encoder_consistency(
    nnsight_lm_encoder: LanguageModel, hf_lm_encoder_and_tokenizer: tuple[AutoModelForMaskedLM, AutoTokenizer]
):
    """Test whether the output from a NNSight-wrapped encoder model is equivalent to the one from a
    LanguageModel-wrapped one operating over text batches"""
    hf_model, tokenizer = hf_lm_encoder_and_tokenizer
    nnsight_model = NNsight(hf_model)
    txt = ["The Eiffel Tower is in the city of [MASK].", "Hello, [MASK]! I am Interpreto."]
    inputs = tokenizer(txt, padding=True, return_tensors="pt")
    inputs_tuple = (inputs["input_ids"], inputs["attention_mask"])
    with nnsight_model.trace(*inputs_tuple):
        logits_nn = nnsight_model.cls.output.save()
    with nnsight_lm_encoder.trace(txt):
        logits_llm = nnsight_lm_encoder.cls.output.save()
    assert torch.allclose(logits_nn, logits_llm)
