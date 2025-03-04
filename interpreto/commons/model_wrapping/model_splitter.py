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

import re
from typing import Any

from nnsight.intervention import Envoy
from nnsight.modeling.language import LanguageModel
from transformers import (
    AutoModel,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto import modeling_auto

from .transformers_classes import (
    get_supported_hf_transformer_autoclasses,
    get_supported_hf_transformer_generation_autoclasses,
    get_supported_hf_transformer_generation_classes,
)


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class ModelSplitter(LanguageModel):
    """Generalized NNsight wrapper around encoder-only, decoder-only and encoder-decoder language models.
    Handles loading of config, tokenizer and model, tokenization, batching and execution.
    Provided splits are used to control early stopping during execution.

    Inputs can be in the form of:
        Prompt: (str)
        Prompts: (List[str])
        Batched prompts: (List[List[str]])
        Tokenized prompt: (Union[List[int], torch.Tensor])
        Tokenized prompts: (Union[List[List[int]], torch.Tensor])
        Direct input: (Dict[str,Any])

    Attributes:
        config (PretrainedConfig): Huggingface config file loaded from repository or checkpoint.
        tokenizer (PreTrainedTokenizer): Tokenizer for LMs.
        model_class (Type): AutoModel type from transformer auto models.
        model (PreTrainedModel): Meta version of underlying auto model.

    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        splits: str | list[str],
        *args: tuple[Any],
        model_class: str | type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        **kwargs,
    ) -> None:
        if isinstance(model_or_repo_id, str):  # Repository ID
            if model_class is None:
                raise InitializationError(
                    "Model class not found.\n"
                    "The model class can be omitted if a pre-loaded model is passed to `model_or_repo_id` "
                    "param.\nIf an HF Hub ID is used, the corresponding autoclass must be specified in `model_type`.\n"
                    "Example: ModelSplitWrapper('bert-base-cased', model_type=AutoModelForMaskedLM, ...)"
                )
            if isinstance(model_class, str):
                supported_autoclasses = get_supported_hf_transformer_autoclasses()
                try:
                    self.model_class: type[AutoModel] = getattr(modeling_auto, model_class)
                except AttributeError:
                    raise InitializationError(
                        f"The specified class {model_class} is not a valid autoclass.\n"
                        f"Supported autoclasses: {', '.join(supported_autoclasses)}"
                    ) from AttributeError
                if model_class not in supported_autoclasses:
                    raise InitializationError(
                        f"The specified autoclass {model_class} is not supported.\n"
                        f"Supported autoclasses: {', '.join(supported_autoclasses)}"
                    )
            else:
                self.model_class: type[AutoModel] = model_class
        self.splits: list[str] = splits if isinstance(splits, list) else [splits]

        # Handles model loading through LanguageModel._load
        super().__init__(
            model_or_repo_id,
            *args,
            config=config,
            tokenizer=tokenizer,  # type: ignore
            **kwargs,
        )
        self._validate_splits()
        self._model: PreTrainedModel
        if self.repo_id is None:
            self.repo_id = self._model.config.name_or_path
        self.generator: Envoy | None
        if self._model.__class__.__name__ not in get_supported_hf_transformer_generation_classes():
            self.generator = None  # type: ignore

    def _validate_splits(self):
        """Validates that specified splits are present in the loaded model."""
        for split in self.splits:
            # Split the path into components, handling list indexing
            components = re.findall(r"([^\.\[\]]+)(?:\[(\d+)\])?", split)[1:]
            current = self._model
            current_path = "model"
            for name, index in components:
                try:
                    # First, try to get the attribute
                    current = getattr(current, name)
                    current_path += f".{name}"
                    # If an index is specified, treat as list/sequence access
                    if index:
                        current = current[int(index)]
                        current_path += f"[{index}]"
                except AttributeError as ex:
                    # Get available submodules for more informative error
                    try:
                        available_submodules = list(current._modules.keys())
                    except Exception:
                        available_submodules = []
                    raise InitializationError(
                        f"The provided splitting point '{split}' is not valid.\n"
                        f"Module {current_path} does not have submodule '{name}'.\n"
                        f"Available submodule names: {', '.join(available_submodules)}.\n"
                        "Use 'model.[...] to define a split based on the structure of your model, or pass an integer "
                        "corresponding to the selected layer to use the output of that layer as splitting point."
                    ) from ex
                except (IndexError, TypeError) as ex:
                    raise InitializationError(
                        f"The provided splitting point '{split}' is not valid.\n"
                        f"Cannot access {current_path} with index '{index}'.\n"
                        "Use 'model.[...] to define a split based on the structure of your model, or pass an integer "
                        "corresponding to the selected layer to use the output of that layer as splitting point."
                    ) from ex
        return True

    def _generate(
        self,
        inputs: BatchEncoding,
        max_new_tokens=1,
        streamer: Any = None,
        **kwargs,
    ):
        if self.generator is None:
            gen_classes = get_supported_hf_transformer_generation_autoclasses()
            raise RuntimeError(
                f"model.generate was called but model class {self._model.__class__.__name__} does not support "
                "generation. Use regular forward passes for inference, or change model_class in the initialization "
                f"to use a generative class. Supported classes: {', '.join(gen_classes)}."
            )
        super()._generate(inputs=inputs, max_new_tokens=max_new_tokens, streamer=streamer, **kwargs)
