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
Sobol attribution method
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations.sobol_aggregation import SobolAggregator
from interpreto.attributions.base import (
    AttributionExplainer,
    MultitaskExplainerMixin,
)
from interpreto.attributions.perturbations.sobol_perturbation import (
    SequenceSamplers,
    SobolIndicesOrders,
    SobolTokenPerturbator,
)
from interpreto.commons.granularity import GranularityLevel


class SobolAttribution(MultitaskExplainerMixin, AttributionExplainer):
    """
    Sobol Attribution method
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        granularity_level: GranularityLevel = GranularityLevel.WORD,
        n_token_perturbations: int = 30,
        sobol_indices_order: SobolIndicesOrders = SobolIndicesOrders.FIRST_ORDER,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        device: torch.device | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the attribution method
            granularity_level (GranularityLevel): The level of granularity for the explanation (e.g., token, word, sentence).
            n_token_perturbations (int): the number of perturbations to generate
            sobol_indices (SobolIndicesOrders): Sobol indices order, either `FIRST_ORDER` or `TOTAL_ORDER`.
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
            device (torch.device): device on which the attribution method will be run
        """
        # TODO : move this in upper class (MaskingExplainer or something)
        replace_token = "[REPLACE]"
        if replace_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([replace_token])
            model.resize_token_embeddings(len(tokenizer))
        replace_token_id = tokenizer.convert_tokens_to_ids(replace_token)
        if isinstance(replace_token_id, list):
            replace_token_id = replace_token_id[0]

        perturbator = SobolTokenPerturbator(
            inputs_embedder=model.get_input_embeddings(),
            granularity_level=granularity_level,
            replace_token_id=replace_token_id,
            n_token_perturbations=n_token_perturbations,
            sobol_indices_order=sobol_indices_order,
            sampler=sampler,
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            perturbator=perturbator,
            aggregator=SobolAggregator(n_token_perturbations=n_token_perturbations),
            batch_size=batch_size,
            usegradient=False,
            granularity_level=granularity_level,
            device=device,
        )
