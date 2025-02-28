"""
Concept Bottleneck Explainer based on Overcomplete concept-encoder-decoder framework.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from interpreto.commons.model_wrapping.model_splitter import ModelSplitterPlaceholder
from interpreto.concepts.base import ConceptBottleneckExplainer
from interpreto.typing import LatentActivation

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


class OvercompleteMethods(NamedTuple):
    """
    Overcomplete concepts encoder decoders classes for dictionary learning.
    TODO: add link to Thomas' lib a bit everywhere
    """

    SAE: type[oc_sae.SAE] = oc_sae.SAE
    TopKSAE: type[oc_sae.SAE] = oc_sae.TopKSAE
    BatchTopKSAE: type[oc_sae.SAE] = oc_sae.BatchTopKSAE
    JumpSAE: type[oc_sae.SAE] = oc_sae.JumpSAE
    NMF: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.NMF
    SemiNMF: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SemiNMF
    ConvexNMF: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.ConvexNMF
    PCA: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SkPCA
    ICA: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SkICA
    KMeans: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SkKMeans
    DictionaryLearning: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SkDictionaryLearning
    SparsePCA: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SkSparsePCA
    SVD: type[oc_opt.BaseOptimDictionaryLearning] = oc_opt.SkSVD


class OvercompleteSAE(ConceptBottleneckExplainer):
    """
    Implementation of a concept explainer based on the Overcomplete SAEs framework.

    Attributes:
        splitted_model (ModelSplitterPlaceholder): Model splitter
        concept_encoder_decoder (oc_sae.SAE): Overcomplete SAE model
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        _differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    _differentiable_concept_encoder = True
    _differentiable_concept_decoder = True

    def __init__(
        self,
        splitted_model: ModelSplitterPlaceholder,
        ConceptEncoderDecoder: type[oc_sae.SAE],
        n_concepts: int,
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
        super().__init__(splitted_model.model)

        self.concept_encoder_decoder = ConceptEncoderDecoder(
            input_shape=self.splitted_model.model.input_to_latent.in_features,
            nb_concepts=n_concepts,
            encoder_module=encoder_module,
            dictionary_params=dictionary_params,
            device=device,
            **kwargs,
        )
        self.fitted = False

    def fit(
        self,
        activations: dict[LatentActivation],
        split: str,
        use_amp: bool = False,
        batch_size: int = 1024,
        criterion: Criterion = mse_criterion,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
            activations (dict[LatentActivation]): The activations to train the concept encoder-decoder on.
            split (str): The split to use for the explanation.
            use_amp (bool): Whether to use automatic mixed precision.
            criterion (Criterion): Loss criterion for the training of the concept encoder-decoder.
            optimizer (torch.optim.Optimizer): Optimizer for the training of the concept encoder-decoder.
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): Learning rate scheduler for the training of the concept encoder-decoder.
            lr (float): Learning rate for the training of the concept encoder-decoder.
            nb_epochs (int): Number of epochs for the training of the concept encoder-decoder.
            clip_grad (float | None): Gradient clipping value for the training of the concept encoder-decoder.
            monitoring (int | None): Monitoring frequency for the training of the concept encoder-decoder.
            device (torch.device | str): Device to use for the training of the concept encoder-decoder.
            max_nan_fallbacks (int | None): Maximum number of fallbacks to use when NaNs are encountered during training. Ignored if use_amp is False.

        Returns:
            log: dictionary with training history.
        """
        inputs = activations[split]
        self.split = split

        dataloader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=True)
        optimizer = torch.optimizer.Adam(self.concept_encoder_decoder.parameters(), lr=lr)

        if use_amp:
            log = oc_sae.train_sae_amp(
                model=self.concept_encoder_decoder,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=None,
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
                scheduler=None,
                nb_epochs=nb_epochs,
                clip_grad=clip_grad,
                monitoring=monitoring,
                device=device,
            )
        return log

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

    Attributes:
        splitted_model (ModelSplitterPlaceholder): Model splitter
        concept_encoder_decoder (oc_opt.BaseOptimDictionaryLearning): Overcomplete optimization module for dictionary learning
        fitted (bool): Whether the model has been fitted
        _differentiable_concept_encoder (bool): Whether the concept encoder is differentiable.
        _differentiable_concept_decoder (bool): Whether the concept decoder is differentiable.
    """

    _differentiable_concept_decoder = True

    def __init__(
        self,
        splitted_model: ModelSplitterPlaceholder,
        ConceptEncoderDecoder: type[oc_opt.BaseOptimDictionaryLearning],
        n_concepts: int,
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
        super().__init__(splitted_model.model)
        self.concept_encoder_decoder = ConceptEncoderDecoder(nb_concepts=n_concepts, device=device, **kwargs)
        self.fitted = False

        if "NMF" in ConceptEncoderDecoder.__name__:
            self._differentiable_concept_encoder = False

    def fit(self, activations: dict[LatentActivation], split: str, **kwargs):
        """
        ConceptEncoderDecoder for explaining a concept.

        Args:
            activations (dict[LatentActivation]): The activations to train the concept encoder-decoder on. Shape: (n_samples, n_features).
            split (str): The split to use for the explanation.
            **kwargs: Additional keyword arguments to pass to the concept encoder-decoder. See the Overcomplete documentation of the provided `ConceptEncoderDecoder` for more details.
        """
        inputs = activations[split]
        self.split = split

        self.concept_encoder_decoder.fit(inputs, **kwargs)
        self.fitted = True
