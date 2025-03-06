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
Concept Bottleneck Explainer based on Overcomplete concept-encoder-decoder framework.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

import torch
from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.base import ConceptBottleneckExplainer
from interpreto.typing import ConceptsActivations, LatentActivations

if TYPE_CHECKING:
    Criterion = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def mse_criterion(
    x: torch.Tensor, x_hat: torch.Tensor, pre_codes: torch.Tensor, codes: torch.Tensor, dictionary: torch.Tensor
) -> torch.Tensor:
    return (x - x_hat).square().mean()


def dead_neurons_reanimation_criterion(
    x: torch.Tensor, x_hat: torch.Tensor, pre_codes: torch.Tensor, codes: torch.Tensor, dictionary: torch.Tensor
) -> torch.Tensor:
    loss = (x - x_hat).square().mean()

    # is dead of shape (k) (nb concepts) and is 1 iif
    # not a single code has fire in the batch
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
    # we push the pre_codes (before relu) towards the positive orthant
    reanim_loss = (pre_codes * is_dead[None, :]).mean()

    loss -= reanim_loss * 1e-3
    return loss


class OvercompleteMethods(Enum):
    """
    Overcomplete concepts encoder-decoder classes for dictionary learning.
    https://github.com/KempnerInstitute/overcomplete/tree/main
    """

    SAE = oc_sae.SAE
    TopKSAE = oc_sae.TopKSAE
    BatchTopKSAE = oc_sae.BatchTopKSAE
    JumpSAE = oc_sae.JumpSAE
    NMF = oc_opt.NMF
    SemiNMF = oc_opt.SemiNMF
    ConvexNMF = oc_opt.ConvexNMF
    PCA = oc_opt.SkPCA
    ICA = oc_opt.SkICA
    KMeans = oc_opt.SkKMeans
    DictionaryLearning = oc_opt.SkDictionaryLearning
    SparsePCA = oc_opt.SkSparsePCA
    SVD = oc_opt.SkSVD


class OvercompleteSAE(ConceptBottleneckExplainer):
    """
    Implementation of a concept explainer based on the Overcomplete SAEs framework.
    https://github.com/KempnerInstitute/overcomplete/tree/main

    Attributes:
        splitted_model (ModelSplitterPlaceholder): Model splitter
        split (str): The split in the model where the concepts are encoded from.
        concept_encoder_decoder (oc_sae.SAE): Overcomplete SAE model
        is_fitted (bool): Whether the model has been fitted
        has_differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        has_differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    has_differentiable_concept_encoder = True
    has_differentiable_concept_decoder = True

    def __init__(
        self,
        splitted_model: ModelSplitterPlaceholder,
        ConceptEncoderDecoder: type[oc_sae.SAE],
        n_concepts: int,
        *,
        encoder_module: nn.Module | str | None = None,
        dictionary_params: dict | None = None,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete SAE framework.

        Args:
            splitted_model (ModelSplitterPlaceholder):The model to apply the explanation on. It should be splitted between at least two parts.
            ConceptEncoderDecoder (OvercompleteMethods): Overcomplete dictionary learning class to use for decomposition.
            n_concepts (int): Number of concepts to explain.
            encoder_module (nn.Module | str | None): Encoder module to use for the concept encoder-decoder.
            dictionary_params (dict | None): Dictionary parameters to use for the concept encoder-decoder.
            device (torch.device | str): Device to use for the concept encoder-decoder.
            **kwargs: Additional keyword arguments to pass to the concept encoder-decoder. See the Overcomplete documentation of the provided `ConceptEncoderDecoder` for more details.
        """
        super().__init__(splitted_model)

        if not issubclass(ConceptEncoderDecoder, oc_sae.SAE):
            raise ValueError(
                "ConceptEncoderDecoder must be a subclass of `overcomplete.sae.SAE`. Use `OvercompleteMethods` instead."
            )

        self.concept_encoder_decoder = ConceptEncoderDecoder(
            input_shape=self.splitted_model.latent_shape,  # TODO: adapt this, it is tricky because it requires to know the split
            nb_concepts=n_concepts,
            encoder_module=encoder_module,
            dictionary_params=dictionary_params,
            device=device,
            **kwargs,
        )
        self.is_fitted = False

    def fit(
        self,
        activations: dict[str, LatentActivations],
        *,
        split: str | None = None,
        use_amp: bool = False,
        batch_size: int = 1024,
        criterion: Criterion = mse_criterion,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr: float = 1e-3,
        nb_epochs: int = 20,
        clip_grad: float | None = None,
        monitoring: int | None = None,
        device: torch.device | str = "cpu",
        max_nan_fallbacks: int | None = 5,
    ) -> dict:
        """
        ConceptEncoderDecoder for explaining a concept.

        Args:
            activations (dict[str, LatentActivations]): the activations to train the concept encoder on.
            split: (str | None): The dataset split to use for training the concept encoder. If None, the model is assumed to be a single-split model. And split is inferred from the keys of the activations dict.
            use_amp (bool): Whether to use automatic mixed precision.
            criterion (Criterion): Loss criterion for the training of the concept encoder-decoder.
            optimizer_class (type[torch.optim.Optimizer]): Optimizer for the training of the concept encoder-decoder.
            lr (float): Learning rate for the training of the concept encoder-decoder.
            nb_epochs (int): Number of epochs for the training of the concept encoder-decoder.
            clip_grad (float | None): Gradient clipping value for the training of the concept encoder-decoder.
            monitoring (int | None): Monitoring frequency for the training of the concept encoder-decoder.
            device (torch.device | str): Device to use for the training of the concept encoder-decoder.
            max_nan_fallbacks (int | None): Maximum number of fallbacks to use when NaNs are encountered during training. Ignored if use_amp is False.

        Returns:
            log: dictionary with training history.
        """
        inputs, self.split = self.verify_activations(activations, split)
        assert len(inputs.shape) == 2, (
            f"Inputs should be a 2D tensor, (batch_size, n_features) but got {inputs.shape}."
        )

        dataloader = DataLoader(TensorDataset(inputs.detach()), batch_size=batch_size, shuffle=True)
        optimizer = optimizer_class(self.concept_encoder_decoder.parameters(), lr=lr)

        if use_amp:
            log = oc_sae.train_sae_amp(
                model=self.concept_encoder_decoder,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                nb_epochs=nb_epochs,
                clip_grad=clip_grad,
                monitoring=monitoring,
                device=device,
                max_nan_fallbacks=max_nan_fallbacks,
            )
        else:
            log = oc_sae.train_sae(
                model=self.concept_encoder_decoder,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                nb_epochs=nb_epochs,
                clip_grad=clip_grad,
                monitoring=monitoring,
                device=device,
            )
        self.is_fitted = True
        return log

    def encode_activations(
        self, activations: LatentActivations | dict[str, LatentActivations], **kwargs
    ) -> ConceptsActivations:
        """
        Encode the given activations using the concept encoder-decoder.

        Args:
            activations (LatentActivations | dict[str, LatentActivations]): The activations to encode.

        Returns:
            ConceptsActivations: The encoded activations.
        """
        assert self.is_fitted, "Concept explainer has not been fitted yet."

        inputs, _ = self.verify_activations(activations)
        inputs = inputs.to(self.concept_encoder_decoder.device)

        # SAEs.encode returns both codes (concepts activations) and pre_codes (before relu)
        _, codes = self.concept_encoder_decoder.encode(inputs, **kwargs)
        return codes

    def to(self, device: torch.device | str):
        """
        Move the concept bottleneck explainer to a new device.

        Args:
            device (torch.device | str): The device to move the explainer to.
        """
        self.concept_encoder_decoder.to(device)

    def cpu(self):
        """
        Move the concept bottleneck explainer to the CPU.
        """
        self.concept_encoder_decoder.cpu()

    def cuda(self, device: int = 0):
        """
        Move the concept bottleneck explainer to the GPU.
        """
        self.concept_encoder_decoder.cuda(device)


class OvercompleteDictionaryLearning(ConceptBottleneckExplainer):
    """
    Implementation of a concept explainer based on the Overcomplete optimization module for dictionary learning.
    https://github.com/KempnerInstitute/overcomplete/tree/main

    Attributes:
        splitted_model (ModelSplitterPlaceholder): Model splitter
        split (str): The split in the model where the concepts are encoded from.
        concept_encoder_decoder (oc_opt.BaseOptimDictionaryLearning): Overcomplete optimization module for dictionary learning
        is_fitted (bool): Whether the model has been fitted
        has_differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        has_differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    has_differentiable_concept_decoder = True

    def __init__(
        self,
        splitted_model: ModelSplitterPlaceholder,
        ConceptEncoderDecoder: type[oc_opt.BaseOptimDictionaryLearning],
        n_concepts: int,
        *,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete optimization module for dictionary learning.

        Args:
            splitted_model (ModelSplitterPlaceholder):The model to apply the explanation on. It should be splitted between at least two parts.
            ConceptEncoderDecoder (OvercompleteMethods): Overcomplete dictionary learning class to use for decomposition.
            n_concepts (int): Number of concepts to explain.
            device (torch.device | str): Device to use for the concept encoder-decoder.
            **kwargs: Additional keyword arguments to pass to the concept encoder-decoder. See the Overcomplete documentation of the provided `ConceptEncoderDecoder` for more details.
        """
        super().__init__(splitted_model)

        if not issubclass(ConceptEncoderDecoder, oc_opt.BaseOptimDictionaryLearning):
            raise ValueError(
                "ConceptEncoderDecoder must be a subclass of `overcomplete.optimization.BaseOptimDictionaryLearning`. Use `OvercompleteMethods` instead."
            )

        if ConceptEncoderDecoder.__name__ == "ConvexNMF":
            # TODO: see if we can support the pgd solver or have an easy way to set parameters
            kwargs["solver"] = "mu"

        self.concept_encoder_decoder = ConceptEncoderDecoder(nb_concepts=n_concepts, device=device, **kwargs)
        self.is_fitted = False

        if "NMF" in ConceptEncoderDecoder.__name__:
            self.has_differentiable_concept_encoder = False

    def fit(self, activations: dict[str, LatentActivations], *, split: str | None = None, **kwargs):
        """
        ConceptEncoderDecoder for explaining a concept.

        Args:
            activations (dict[str, LatentActivations]): the activations to train the concept encoder on. Shape: (n_samples, n_features).
            split: (str | None): The dataset split to use for training the concept encoder. If None, the model is assumed to be a single-split model. And split is inferred from the keys of the activations dict.
            **kwargs: Additional keyword arguments to pass to the concept encoder-decoder. See the Overcomplete documentation of the provided `ConceptEncoderDecoder` for more details.
        """
        inputs, self.split = self.verify_activations(activations, split)
        assert len(inputs.shape) == 2, (
            f"Inputs should be a 2D tensor, (batch_size, n_features) but got {inputs.shape}."
        )

        self.concept_encoder_decoder.fit(inputs, **kwargs)
        self.is_fitted = True
