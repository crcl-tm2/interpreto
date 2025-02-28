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

import json
from typing import Any

import torch
from nnsight.modeling.mixins import RemoteableMixin
from torch.nn.modules import Module
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.auto import modeling_auto
from typing_extensions import Self


class LanguageModelEncoder(RemoteableMixin):
    """LanguageModels are NNsight wrappers around transformers language models. We create an Encoder version of LanguageModel.

    Inputs can be in the form of:
        Prompt: (str)
        Prompts: (List[str])
        Batched prompts: (List[List[str]])
        Tokenized prompt: (Union[List[int], torch.Tensor])
        Tokenized prompts: (Union[List[List[int]], torch.Tensor])
        Direct input: (Dict[str,Any])

    If using a custom model, you also need to provide the tokenizer like ``LanguageModel(custom_model, tokenizer=tokenizer)``

    Attributes:
        config (PretrainedConfig): Huggingface config file loaded from repository or checkpoint.
        tokenizer (PreTrainedTokenizer): Tokenizer for LMs.
        automodel (Type): AutoModel type from transformer auto models.
        model (PreTrainedModel): Meta version of underlying auto model.

    """

    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        *args,
        config: PretrainedConfig | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        automodel: type[
            AutoModel
        ],  ####not by default. Choose: type[AutoModel] = AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification
        **kwargs,
    ) -> None:
        self.automodel = automodel if not isinstance(automodel, str) else getattr(modeling_auto, automodel)

        self.config = config
        self.tokenizer = tokenizer
        self.repo_id: str = None

        super().__init__(*args, **kwargs)

    def _load_config(self, repo_id: str, **kwargs):
        if self.config is None:
            self.config = AutoConfig.from_pretrained(repo_id, **kwargs)

    def _load_tokenizer(self, repo_id: str, **kwargs):
        if self.tokenizer is None:
            if "padding_side" not in kwargs:
                kwargs["padding_side"] = "right"
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, config=self.config, **kwargs)

            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_meta(
        self,
        repo_id: str,
        tokenizer_kwargs: dict[str, Any] | None = {},
        **kwargs,
    ) -> Module:
        self.repo_id = repo_id

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)

        model = self.automodel.from_config(self.config, trust_remote_code=True)

        return model

    def _load(
        self,
        repo_id: str,
        tokenizer_kwargs: dict[str, Any] | None = {},
        **kwargs,
    ) -> PreTrainedModel:
        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)

        model = self.automodel.from_pretrained(repo_id, config=self.config, **kwargs)

        return model

    def _tokenize(
        self,
        inputs: str | list[str] | list[list[str]] | list[int] | list[list[int]] | torch.Tensor | dict[str, Any],
        **kwargs,
    ):
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], int)):
            inputs = [inputs]

        if isinstance(inputs, torch.Tensor) and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        if not isinstance(inputs[0], str):
            inputs = [{"input_ids": ids} for ids in inputs]
            return self.tokenizer.pad(inputs, return_tensors="pt", **kwargs)

        return self.tokenizer(inputs, return_tensors="pt", padding=True, **kwargs)

    def _prepare_input(
        self,
        *inputs: tuple[
            str
            | list[str]
            | list[list[str]]
            | list[int]
            | list[list[int]]
            | torch.Tensor
            | list[torch.Tensor]
            | dict[str, Any]
            | BatchEncoding
        ],
        input_ids: list[int] | list[list[int]] | torch.Tensor | list[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[BatchEncoding, int]:  ###TODO: see ouput type
        if input_ids is not None:
            assert len(inputs) == 0

            inputs = (input_ids,)

        assert len(inputs) == 1

        inputs = inputs[0]

        if isinstance(inputs, dict):
            inputs = BatchEncoding(inputs)
        elif isinstance(inputs, BatchEncoding):
            pass
        else:
            inputs = self._tokenize(inputs, **kwargs)

        return (inputs,), len(inputs["input_ids"])

    def _batch(
        self,
        batched_inputs: tuple[tuple[BatchEncoding], dict[str, Any]] | None,
        input: BatchEncoding,
    ) -> tuple[dict[str, Any]]:  ###TODO: see ouput type
        if batched_inputs is None:
            return (input,)

        batched_inputs = batched_inputs[0][0]

        attention_mask = batched_inputs["attention_mask"]
        batched_inputs = [
            {"input_ids": ids}
            for ids in [
                *batched_inputs["input_ids"].tolist(),
                *input["input_ids"].tolist(),
            ]
        ]
        batched_inputs = self.tokenizer.pad(batched_inputs, return_tensors="pt")

        if self.tokenizer.padding_side == "left":
            batched_inputs["attention_mask"][: attention_mask.shape[0], -attention_mask.shape[1] :] = attention_mask

        else:
            batched_inputs["attention_mask"][: attention_mask.shape[0], : attention_mask.shape[1]] = attention_mask

        return (batched_inputs,)

    def _execute(self, inputs: BatchEncoding, **kwargs) -> Any:
        inputs = inputs.to(self.device)

        return self._model(
            **inputs,
            **kwargs,
        )

    def _remoteable_model_key(self) -> str:
        return json.dumps(
            {"repo_id": self.repo_id}  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:
        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        return LanguageModelEncoder(repo_id, **kwargs)
