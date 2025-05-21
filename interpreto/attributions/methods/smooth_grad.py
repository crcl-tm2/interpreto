"""
SmoothGrad method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations import MeanAggregator
from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.attributions.perturbations import GaussianNoisePerturbator
from interpreto.commons.model_wrapping.inference_wrapper import InferenceModes


class SmoothGrad(MultitaskExplainerMixin, AttributionExplainer):
    """
    SmoothGrad method
    """

    use_gradient = True

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        device: torch.device | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_interpolations: int = 10,
        noise_level: float = 0.1,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the attribution method
            device (torch.device): device on which the attribution method will be run
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
            n_interpolations (int): the number of interpolations to generate
            noise_level (float): standard deviation of the Gaussian noise to add to the inputs
        """
        perturbator = GaussianNoisePerturbator(
            inputs_embedder=model.get_input_embeddings(), n_perturbations=n_interpolations, std=noise_level
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            perturbator=perturbator,
            aggregator=MeanAggregator(),
            inference_mode=inference_mode,
        )
