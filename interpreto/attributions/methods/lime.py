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
LIME attribution method
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.aggregations.linear_regression_aggregation import (
    DistanceFunctions,
    Kernels,
    LinearRegressionAggregator,
    default_kernel_width_fn,
)
from interpreto.attributions.base import InferenceExplainer
from interpreto.attributions.perturbations.base import RandomMaskedTokenPerturbator
from interpreto.commons.model_wrapping.inference_wrapper import ClassificationInferenceWrapper


class Lime(InferenceExplainer):
    """
    Sobol Attribution method
    """

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer | None = None,
        n_token_perturbations: int = 30,
        granularity_level: str = "token",
        baseline: str = "[MASK]",
        batch_size: int = 1,
        distance_function: DistanceFunctions = DistanceFunctions.COSINE,
        similarity_kernel: Callable = Kernels.EXPONENTIAL,
        kernel_width: float | Callable = default_kernel_width_fn,
        device: torch.device | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (Any): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            n_perturbations (int): the number of perturbations to generate
            granularity_level (str): granularity level of the perturbations (token, word, sentence, etc.)
            baseline (str): replacement token (e.g. “[MASK]”)
            batch_size (int): batch size for the attribution method
            device (torch.device): device on which the attribution method will be run
        """
        if tokenizer is None:
            raise ValueError(
                "Tokenizer must be provided for Sobol token attribution, tensor attributions are not supported yet."
            )

        perturbator = RandomMaskedTokenPerturbator(
            tokenizer=tokenizer,
            inputs_embedder=model.get_input_embeddings(),
            n_token_perturbations=n_token_perturbations,
            granularity_level=granularity_level,
            baseline=baseline,
            device=device,
        )

        aggregator = LinearRegressionAggregator(
            distance_function=distance_function,
            similarity_kernel=similarity_kernel,
            kernel_width=kernel_width,
        )

        super().__init__(
            perturbation=perturbator,
            inference_wrapper=ClassificationInferenceWrapper(model=model, batch_size=batch_size, device=device),
            aggregation=aggregator,
            device=device,
        )
