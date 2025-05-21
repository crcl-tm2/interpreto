"""
Saliency method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.commons.model_wrapping.inference_wrapper import InferenceModes


class Saliency(MultitaskExplainerMixin, AttributionExplainer):
    """
    Saliency method
    """

    use_gradient = True

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        device: torch.device | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
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
        """
        perturbator = None

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            perturbator=perturbator,
            aggregator=None,
            inference_mode=inference_mode,
        )
