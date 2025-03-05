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

from collections.abc import Sequence
from collections.abc import Sequence as SequenceABC
from typing import Any

import torch
from nnsight.intervention import Envoy
from nnsight.intervention.graph import InterventionProxy
from nnsight.modeling.language import LanguageModel
from nnsight.util import fetch_attr
from transformers import (
    AutoModel,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto import modeling_auto

from ...typing import LatentActivations
from .splitting_utils import get_layer_by_idx, sort_paths, validate_path, walk_modules
from .transformers_classes import (
    get_supported_hf_transformer_autoclasses,
    get_supported_hf_transformer_generation_autoclasses,
    get_supported_hf_transformer_generation_classes,
)


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class ModelSplitter(LanguageModel):
    """Generalized NNsight.LanguageModel wrapper around encoder-only, decoder-only and encoder-decoder language models.
    Handles splitting model at specified locations and activation extraction.

    Inputs can be in the form of:
        * One (`str`) or more (`list[str]`) prompts, including batched prompts (`list[list[str]]`).
        * One (`list[int] or torch.Tensor`) or more (`list[list[int]] or torch.Tensor`) tokenized prompts.
        * Direct model inputs: (`dic[str,Any]`)

    Attributes:
        model_autoclass (type): The [AutoClass](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing)
            corresponding to the loaded model type.
        splits (list[str]): Getter/setters for model paths corresponding to split points inside the loaded model.
            Automatically handle validation, sorting and resolving int paths to strings.
        repo_id (str): Either the model id in the HF Hub, or the path from which the model was loaded.
        generator (nnsight.Envoy | None): If the model is generative, a generator is provided to handle multi-step
            inference. None for encoder-only models.
        _model (transformers.PreTrainedModel): Huggingface transformers model wrapped by NNSight.
        _model_paths (list[str]): List of cached valid paths inside `_model`, used to validate `splits`.
        _splits (list[str]): List of splits, should be accessed with getter/setter.

    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        splits: str | int | Sequence[str | int],
        *args: tuple[Any],
        model_autoclass: str | type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize a ModelSplitter object.

        Args:
            model_or_repo_id (str | transformers.PreTrainedModel): One of:

                * A `str` corresponding to the ID of the model that should be loaded from the HF Hub.
                * A `str` corresponding to the local path of a folder containing a compatible checkpoint.
                * A preloaded `transformers.PreTrainedModel` object.
                If a string is provided, a model_autoclass should also be provided.
            splits (str | Sequence[str] | int | Sequence[int]): One or more to split locations inside the model.
                Either the path is provided explicitly (`str`), or an `int` is used as shorthand for splitting at
                the n-th layer. Example: `splits='cls.predictions.transform.LayerNorm'` correspond to a split
                after the LayerNorm layer in the MLM head (assuming a `BertForMaskedLM` model in input).
            model_autoclass (Type): Huggingface [AutoClass](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing)
                corresponding to the desired type of model (e.g. `AutoModelForSequenceClassification`).

                :warning: `model_autoclass` **must be defined** if `model_or_repo_id` is `str`, since the the model class
                    cannot be known otherwise.
            config (PretrainedConfig): Custom configuration for the loaded model.
                If not specified, it will be instantiated with the default configuration for the model.
            tokenizer (PreTrainedTokenizer): Custom tokenizer for the loaded model.
                If not specified, it will be instantiated with the default tokenizer for the model.
        """
        if isinstance(model_or_repo_id, str):  # Repository ID
            if model_autoclass is None:
                raise InitializationError(
                    "Model autoclass not found.\n"
                    "The model class can be omitted if a pre-loaded model is passed to `model_or_repo_id` "
                    "param.\nIf an HF Hub ID is used, the corresponding autoclass must be specified in `model_type`.\n"
                    "Example: ModelSplitter('bert-base-cased', model_type=AutoModelForMaskedLM, ...)"
                )
            if isinstance(model_autoclass, str):
                supported_autoclasses = get_supported_hf_transformer_autoclasses()
                try:
                    self.model_autoclass: type[AutoModel] = getattr(modeling_auto, model_autoclass)
                except AttributeError:
                    raise InitializationError(
                        f"The specified class {model_autoclass} is not a valid autoclass.\n"
                        f"Supported autoclasses: {', '.join(supported_autoclasses)}"
                    ) from AttributeError
                if model_autoclass not in supported_autoclasses:
                    raise InitializationError(
                        f"The specified autoclass {model_autoclass} is not supported.\n"
                        f"Supported autoclasses: {', '.join(supported_autoclasses)}"
                    )
            else:
                self.model_autoclass: type[AutoModel] = model_autoclass

        # Handles model loading through LanguageModel._load
        super().__init__(
            model_or_repo_id,
            *args,
            config=config,
            tokenizer=tokenizer,  # type: ignore
            automodel=self.model_autoclass,
            **kwargs,
        )
        self._model_paths = list(walk_modules(self._model))
        self.splits = splits
        self._model: PreTrainedModel
        if self.repo_id is None:
            self.repo_id = self._model.config.name_or_path
        self.generator: Envoy | None
        if self._model.__class__.__name__ not in get_supported_hf_transformer_generation_classes():
            self.generator = None  # type: ignore

    @property
    def splits(self) -> list[str]:
        return self._splits

    @splits.setter
    def splits(self, splits: str | int | Sequence[str | int]):
        """Splits are automatically validated and sorted upon setting"""
        pre_conversion_splits: Sequence[str | int] = splits if isinstance(splits, Sequence) else [splits]
        post_conversion_splits: list[str] = []
        for split in pre_conversion_splits:
            # Handle conversion of layer idx to full path
            if isinstance(split, int):
                str_split = get_layer_by_idx(split, model_paths=self._model_paths)
            else:
                str_split = split
            post_conversion_splits.append(str_split)

            # Validate whether the split exists in the model
            validate_path(self._model, str_split)

        # Sort splits to match execution order
        self._splits: list[str] = sort_paths(post_conversion_splits, model_paths=self._model_paths)

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
                "generation. Use regular forward passes for inference, or change model_autoclass in the initialization "
                f"to use a generative class. Supported classes: {', '.join(gen_classes)}."
            )
        super()._generate(inputs=inputs, max_new_tokens=max_new_tokens, streamer=streamer, **kwargs)

    def get_activations(self, inputs: str | list[str] | BatchEncoding, **kwargs) -> dict[str, LatentActivations]:
        """Get intermediate activations for all model splits

        Args:
            inputs (str | list[str] | BatchEncoding): Inputs to the model forward pass before or after tokenzation.

        Returns:
            Dictionary having one key, value pair for each split point defined for the model. Keys correspond to split
                names in `self.splits`, while values correspond to the extracted activations for the split point for the
                given `inputs`.
        """
        # TODO: Extend to generation settings
        activations = {}
        if not self.splits:
            raise RuntimeError(
                "No splits are currently defined for the model. Please set splits before calling get_activations."
            )

        # Compute activations
        with self.trace(inputs, **kwargs):
            for idx, split in enumerate(self.splits):
                curr_module: Envoy = fetch_attr(self, split)
                # Handle case in which module has .output attribute, and .nns_output gets overridden instead
                if hasattr(curr_module, "nns_output"):
                    activations[split] = curr_module.nns_output.save()
                else:
                    activations[split] = curr_module.output.save()
                if idx == len(self.splits) - 1:
                    # Early stopping at the last splitting layer
                    curr_module.output.stop()

        # Validate that activations have the expected type
        for layer, act in activations.items():
            if not isinstance(act, torch.Tensor):
                # Handles case in which activations are wrapped by a tuple
                if isinstance(act, (SequenceABC | InterventionProxy)) and isinstance(act[0], torch.Tensor):
                    activations[layer] = act[0]
                else:
                    raise RuntimeError(
                        f"Invalid output for layer '{layer}'. Expected torch.Tensor activation, got {type(act)}: {act}"
                    )
        return activations
