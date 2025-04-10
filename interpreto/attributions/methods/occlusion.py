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

"""
Occlusion attribution method
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.aggregations.base import MaskwiseMeanAggregator
from interpreto.attributions.base import (
    AttributionExplainer,
    ClassificationAttributionExplainer,
    GenerationAttributionExplainer,
)
from interpreto.attributions.perturbations.base import OcclusionPerturbator
from interpreto.commons.granularity import GranularityLevel


class OcclusionExplainer(AttributionExplainer):
    def __new__(cls, model, **kwargs):
        if cls != OcclusionExplainer:
            return super().__new__(cls)
        if model.__class__.__name__.endswith("ForSequenceClassification"):
            return ClassificationOcclusionExplainer.__new__(ClassificationOcclusionExplainer, model, **kwargs)
        elif model.__class__.__name__.endswith("ForCausalLM"):
            return GenerationOcclusionExplainer.__new__(GenerationOcclusionExplainer, model, **kwargs)
        raise NotImplementedError(
            "Model type not supported for OcclusionExplainer. Use an AutoModelForSequenceClassification or AutoModelForCausalLM model."
        )

    def __init__(
        self,
        model: Any,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        granularity_level: GranularityLevel = GranularityLevel.WORD,
        device: torch.device | None = None,
    ):
        replace_token = "[REPLACE]"
        if replace_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([replace_token])
            model.resize_token_embeddings(len(tokenizer))
        replace_token_id = tokenizer.convert_tokens_to_ids(replace_token)

        super().__init__(
            tokenizer=tokenizer,
            inference_wrapper=self._associated_inference_wrapper(model, batch_size=batch_size, device=device),
            perturbator=OcclusionPerturbator(granularity_level=granularity_level, replace_token_id=replace_token_id),  # type: ignore
            aggregator=MaskwiseMeanAggregator(),
            usegradient=False,
            granularity_level=granularity_level,
        )


class ClassificationOcclusionExplainer(OcclusionExplainer, ClassificationAttributionExplainer): ...


class GenerationOcclusionExplainer(OcclusionExplainer, GenerationAttributionExplainer): ...


# class OcclusionExplainerFactory:
#     def __new__(
#         cls,
#         model: Any,
#         batch_size: int,
#         tokenizer: PreTrainedTokenizer,
#         granularity_level: GranularityLevel = GranularityLevel.WORD,
#         device: torch.device | None = None,
#     ):
#         if model.__class__.__name__.endswith(
#             "ForSequenceClassification"
#         ):  # TODO: est ce que on supporte aussi d'autre modele que huggingface pour la classification?
#             return ClassificationOcclusionExplainer(
#                 model=model,
#                 batch_size=batch_size,
#                 tokenizer=tokenizer,
#                 granularity_level=granularity_level,
#                 device=device,
#             )
#         elif model.__class__.__name__.endswith("ForCausalLM"):
#             return GenerationOcclusionExplainer(
#                 model=model,
#                 batch_size=batch_size,
#                 tokenizer=tokenizer,
#                 granularity_level=granularity_level,
#                 device=device,
#             )
#         else:
#             raise NotImplementedError(
#                 "Model type not supported for OcclusionExplainer. Use an AutoModelForSequenceClassification or AutoModelForCausalLM model."
#             )
