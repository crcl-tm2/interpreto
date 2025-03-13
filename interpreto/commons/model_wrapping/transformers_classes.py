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

from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

SUPPORTED_HF_TRANSFORMERS_AUTOCLASSES: list[type] = [
    # Decoder-only generative LMs, e.g. GPT-2
    # Produces transformers.modeling_outputs.CausalLMOutput
    AutoModelForCausalLM,
    # Encoder-decoder generative LMs, e.g. T5
    # Produces transformers.modeling_outputs.Seq2SeqLMOutput
    AutoModelForSeq2SeqLM,
    # Encoder-only LMs with LM head, e.g. BERT for MLM pretraining
    # Produces transformers.modeling_outputs.MaskedLMOutput
    AutoModelForMaskedLM,
    # Encoder-only LMs with classification head, e.g. BERT for sentiment analysis
    # Produces transformers.modeling_outputs.SequenceClassifierOutput
    AutoModelForSequenceClassification,
    # Encoder-only LMs for extractive QA, returning start/end context indices
    # Produces transformers.modeling_outputs.QuestionAnsweringModelOutput
    AutoModelForQuestionAnswering,
    # Encoder-only LMs for e.g. tagging, logits for all input tokens are used
    # Produces transformers.modeling_outputs.TokenClassifierOutput
    AutoModelForTokenClassification,
    # Encoder-only LMs for multiple-choice QA
    # Produces transformers.modeling_outputs.MultipleChoiceModelOutput
    AutoModelForMultipleChoice,
    # AutoModelForImageTextToText,
    # AutoModelForVisualQuestionAnswering,
    # AutoModelForSpeechSeq2Seq,
]

HF_AUTOCLASSES_WITH_GENERATE = [AutoModelForCausalLM, AutoModelForSeq2SeqLM]


def get_supported_hf_transformer_autoclasses() -> set[str]:
    return {autocls.__name__ for autocls in SUPPORTED_HF_TRANSFORMERS_AUTOCLASSES}


def get_supported_hf_transformer_classes(autoclasses: list[type] | None = None) -> set[str]:
    def tuple_dic_from_dic(dic):
        return {(k, v): [] for k, v in dic.items()}

    if autoclasses is None:
        autoclasses = SUPPORTED_HF_TRANSFORMERS_AUTOCLASSES
    model_types = {
        (k, v) for autocls in autoclasses for k, v in tuple_dic_from_dic(autocls._model_mapping._model_mapping)
    }
    model_types = sorted(model_types, key=lambda k: k[0])
    _, model_classes = zip(*model_types, strict=False)
    return set(model_classes)


def get_supported_hf_transformer_generation_autoclasses() -> set[str]:
    return {autocls.__name__ for autocls in HF_AUTOCLASSES_WITH_GENERATE}


def get_supported_hf_transformer_generation_classes() -> set[str]:
    return get_supported_hf_transformer_classes(HF_AUTOCLASSES_WITH_GENERATE)
