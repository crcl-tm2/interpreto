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
from nnsight import NNsight
from pytest import fixture
from transformers import AutoModelForMaskedLM, AutoTokenizer


@fixture
def encoder_lm():
    model = AutoModelForMaskedLM.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", attn_implementation="eager")
    model.requires_grad_(False)
    model.eval()
    return model


def test_attn_output_match_nnsight_hf_encoder_lm(encoder_lm: AutoModelForMaskedLM):
    """Tests whether the extracted output attentions from NNsight match the output attentions from Huggingface.

    Args:
        encoder_lm (AutoModelForMaskedLM): A toy encoder-only transformers model.
    """
    nn = NNsight(encoder_lm)
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    num_layers = len(nn.bert.encoder.layer)

    txt = "Hello, my dog is cute"
    inputs = tokenizer(txt, return_tensors="pt")
    inputs_tuple = (inputs["input_ids"], inputs["attention_mask"])

    attn_dic = {}

    with nn.trace(*inputs_tuple, invoker_kwargs={"output_attentions": True}):
        for i in range(num_layers):
            attn_dic[i] = nn.bert.encoder.layer[i].attention.self.output[1].save()
        nn.bert.encoder.layer[num_layers - 1].attention.self.output.stop()

    print("Attn dict keys:", attn_dic.keys())
    print("One attn entry shape:", attn_dic[0].shape)

    # Extract heads without nnsight
    out = encoder_lm(**inputs, output_attentions=True)

    for i in range(num_layers):
        assert torch.allclose(out.attentions[i], attn_dic[i])
