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
"""Concept encoder/decoder models wrapping scikit-learn components."""

from __future__ import annotations

import torch
from sklearn.decomposition import NMF, PCA, FastICA, TruncatedSVD
from torch import nn


class _BaseEncoderDecoder(nn.Module):
    """Base class for all sklearn concept wrappers.

    Attributes
    ----------
    nb_concepts : int
        Number of concepts handled by the model.
    fitted : bool
        Whether the model has been fitted.
    device : torch.device
        Device on which computations are performed.

    Methods
    -------
    fit(activations)
        Learn the concept representation from activations.
    encode(x)
        Convert activations ``x`` into concepts.
    decode(z)
        Reconstruct activations from concept representation ``z``.
    """

    def __init__(self, nb_concepts: int, *, fitted: bool = False, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.nb_concepts = nb_concepts
        self.fitted = fitted
        self.device = torch.device(device)
        self.to(self.device)

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before use.")

    def fit(self, *args, **kwargs) -> None:  # noqa: D401 - documentation inherited
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


__all__ = [
    "IdentityEncoderDecoder",
    "ICAEncoderDecoder",
    "NMFEncoderDecoder",
    "PCAEncoderDecoder",
    "SVDEncoderDecoder",
]


class IdentityEncoderDecoder(_BaseEncoderDecoder):
    """Identity concept model returning the input as concepts."""

    def __init__(self, nb_concepts: int, *, device: torch.device | str = "cpu") -> None:
        super().__init__(nb_concepts, fitted=True, device=device)

    def fit(self, *args, **kwargs) -> None:  # noqa: D401 - simple implementation
        """Identity model does not need fitting."""
        raise NotImplementedError("Identity concept model does not require fitting.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return x.to(self.device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return z.to(self.device)


class ICAEncoderDecoder(_BaseEncoderDecoder):
    """Independent Component Analysis concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.scale = nn.Linear(input_size, input_size, bias=True)
        self.project = nn.Linear(input_size, nb_concepts, bias=False)
        self.reconstruct = nn.Linear(nb_concepts, input_size, bias=True)
        self.to(self.device)

    def fit(self, activations: torch.Tensor) -> None:
        ica = FastICA(n_components=self.nb_concepts, random_state=self.random_state, max_iter=500)
        ica.fit(activations.detach().cpu().numpy())

        self.scale.weight.data = torch.eye(self.input_size, device=self.device)
        self.scale.bias.data = -torch.as_tensor(ica.mean_, dtype=torch.float32, device=self.device)
        self.project.weight.data = torch.as_tensor(ica.components_, dtype=torch.float32, device=self.device)
        self.reconstruct.weight.data = torch.as_tensor(ica.mixing_, dtype=torch.float32, device=self.device)
        self.reconstruct.bias.data = torch.as_tensor(ica.mean_, dtype=torch.float32, device=self.device)
        self.fitted = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        x = x.to(self.device)
        return self.project(self.scale(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return self.reconstruct(z.to(self.device))


class NMFEncoderDecoder(_BaseEncoderDecoder):
    """Non-negative Matrix Factorisation concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.reconstruct = nn.Linear(nb_concepts, input_size, bias=False)
        self.nmf = NMF(n_components=nb_concepts, random_state=random_state, max_iter=500)
        self.to(self.device)

    def fit(self, activations: torch.Tensor) -> None:
        if (activations < 0).any():
            raise ValueError("NMF only works with non-negative data.")
        self.nmf.fit(activations.detach().cpu().numpy())
        self.reconstruct.weight.data = torch.as_tensor(self.nmf.components_.T, dtype=torch.float32, device=self.device)
        self.fitted = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        concepts = self.nmf.transform(x.detach().cpu().numpy())
        return torch.as_tensor(concepts, dtype=torch.float32, device=self.device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return self.reconstruct(z.to(self.device))


class PCAEncoderDecoder(_BaseEncoderDecoder):
    """Principal Component Analysis concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.scale = nn.Linear(input_size, input_size, bias=True)
        self.project = nn.Linear(input_size, nb_concepts, bias=False)
        self.reconstruct = nn.Linear(nb_concepts, input_size, bias=True)
        self.to(self.device)

    def fit(self, activations: torch.Tensor) -> None:
        pca = PCA(n_components=self.nb_concepts, random_state=self.random_state)
        pca.fit(activations.detach().cpu().numpy())
        self.scale.weight.data = torch.eye(self.input_size, device=self.device)
        self.scale.bias.data = -torch.as_tensor(pca.mean_, dtype=torch.float32, device=self.device)
        self.project.weight.data = torch.as_tensor(pca.components_, dtype=torch.float32, device=self.device)
        self.reconstruct.weight.data = torch.as_tensor(pca.components_.T, dtype=torch.float32, device=self.device)
        self.reconstruct.bias.data = torch.as_tensor(pca.mean_, dtype=torch.float32, device=self.device)
        self.fitted = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        x = x.to(self.device)
        return self.project(self.scale(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return self.reconstruct(z.to(self.device))


class SVDEncoderDecoder(_BaseEncoderDecoder):
    """Singular Value Decomposition concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.project = nn.Linear(input_size, nb_concepts, bias=False)
        self.reconstruct = nn.Linear(nb_concepts, input_size, bias=False)
        self.to(self.device)

    def fit(self, activations: torch.Tensor) -> None:
        svd = TruncatedSVD(n_components=self.nb_concepts, random_state=self.random_state)
        svd.fit(activations.detach().cpu().numpy())
        self.project.weight.data = torch.as_tensor(svd.components_, dtype=torch.float32, device=self.device)
        self.reconstruct.weight.data = torch.as_tensor(svd.components_.T, dtype=torch.float32, device=self.device)
        self.fitted = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return self.project(x.to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted()
        return self.reconstruct(z.to(self.device))
