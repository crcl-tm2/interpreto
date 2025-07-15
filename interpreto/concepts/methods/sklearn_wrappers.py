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
This file contains classes for concept decomposition.

The encode and decode functions were inspired by the transform and inverse_transform
functions of sklearn's decomposition classes.
"""

from abc import ABC, abstractmethod
import os
from typing import Optional, List

import numpy as np
import pickle
import scipy
import torch
from torch import nn

from sklearn.decomposition import FastICA, NMF, PCA, TruncatedSVD


class ConceptEncoderDecoder(nn.Module, ABC):
    """
    Class for concept encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    other_instances
        Other instances to aggregate. If provided, A is not used.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """

    def __init__(
        self,
        A: torch.tensor = None,
        n_concepts: Optional[int] = None,
        n_features: Optional[int] = None,
        other_instances: Optional[List["ConceptEncoderDecoder"]] = None,
        force_recompute: bool = False,
    ):
        super().__init__()
        assert not (A is not None and other_instances is not None), (
            "One and only one of either A or other_instances must be provided."
        )

        if other_instances is not None:
            self.n_concepts = sum([enc_dec.n_concepts for enc_dec in other_instances])
            self.n_features = other_instances[0].n_features
        else:
            assert n_concepts is not None, "Concepts decomposition cannot be computed without n_concepts."
            self.n_concepts = n_concepts
            self.n_features = n_features if n_features is not None else A.shape[1]

    def get_weights(
        self,
        A: torch.tensor = None,
        other_instances: Optional[List["ConceptEncoderDecoder"]] = None,
        force_recompute: bool = False,
    ):
        if self.save_path is not None and not force_recompute:
            try:
                self.get_weights = self._get_weights_from_loading()
                return
            except FileNotFoundError:
                pass

        if other_instances is not None:
            self.get_weights = self._get_weights_from_aggregating(other_instances)
        else:
            assert A is not None, "Concepts decomposition cannot be computed without input data."
            self.get_weights = self._get_weights_from_decomposing(A)

    @abstractmethod
    def _get_weights_from_loading(self):
        raise NotImplementedError

    @abstractmethod
    def _get_weights_from_decomposing(self, A: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def _get_weights_from_aggregating(self, other_instances: List["ConceptEncoderDecoder"]):
        raise NotImplementedError

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encode(x)
        x = self.decode(x_encoded)
        return x

    @property
    @abstractmethod
    def is_differentiable(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def concepts_base(self):
        raise NotImplementedError


class IdentityEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for Identity encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """

    def __init__(
        self,
        A: torch.tensor = None,
        n_concepts: Optional[int] = None,
        n_features: Optional[int] = None,
        other_instances: Optional[List["ConceptEncoderDecoder"]] = None,
        force_recompute: bool = False,
    ):
        super().__init__(
            A=A,
            n_concepts=n_concepts,
            n_features=n_features,
            other_instances=other_instances,
            force_recompute=force_recompute,
        )

        assert self.n_concepts == self.n_features, "IdentityEncoderDecoder requires n_concepts == n_features."

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return x_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def is_differentiable(self):
        return True

    @property
    def concepts_base(self):
        return torch.eye(self.n_features)


class ICAEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for ICA encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """

    def __init__(
        self,
        A: torch.tensor = None,
        n_concepts: Optional[int] = None,
        n_features: Optional[int] = None,
        other_instances: Optional[List["ICAEncoderDecoder"]] = None,
        random_state: int = 0,
        save_path: Optional[str] = None,
        force_recompute: bool = False,
    ):
        super().__init__(
            A=A,
            n_concepts=n_concepts,
            n_features=n_features,
            other_instances=other_instances,
        )

        # initialize method
        self.save_path = os.path.join(save_path, "weights.pt")
        self.random_state = random_state

        # initialize ICA layers
        self.scale = nn.Linear(self.n_features, self.n_features, bias=True)
        self.project = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=True)

        self.get_weights(A=A, other_instances=other_instances, force_recompute=force_recompute)

    def _get_weights_from_loading(self):
        self.load_state_dict(torch.load(self.save_path))

    def _get_weights_from_decomposing(self, A: torch.Tensor):
        self.max_iter = 500

        A = A.numpy()
        ica = FastICA(n_components=self.n_concepts, random_state=self.random_state, max_iter=self.max_iter)
        ica.fit(A)

        # set layers weights with ICA components
        # set encode scaling
        # X - ica.means_
        self.scale.weight.data = torch.eye(self.n_features)
        self.scale.bias.data = -torch.tensor(ica.mean_)

        # set encode projection
        # X_scaled @ ica.components_.T
        self.project.weight.data = torch.tensor(ica.components_)

        # set decode
        # X @ ica.mixing_.T + ica.means_
        self.reconstruct.weight.data = torch.tensor(ica.mixing_)
        self.reconstruct.bias.data = torch.tensor(ica.mean_)

        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def _get_weights_from_aggregating(self, other_instances: List["ICAEncoderDecoder"]):
        """
        Create a new ICAEncoderDecoder by aggregating a list of ICAEncoderDecoder.
        The number of concepts is the sum of the number of concepts of the ICAEncoderDecoder.
        The components are concatenated.

        Parameters
        ----------
        other_instances
            The list of ICAEncoderDecoder to aggregate.

        Returns
        -------
        ICAEncoderDecoder
            The aggregated ICAEncoderDecoder.
        """
        # inherit the constant attributes from the first element of the list
        self.scale.weight.data = other_instances[0].scale.weight.data
        means = torch.cat([encoder_decoder.scale.weight.data for encoder_decoder in other_instances], dim=1).mean(
            dim=1
        )
        self.scale.bias.data = means
        self.reconstruct.bias.data = -means

        # concatenate encoder weights
        self.project.weight.data = torch.cat(
            [encoder_decoder.project.weight.data for encoder_decoder in other_instances], dim=0
        )

        # concatenate decoder weights
        self.reconstruct.weight.data = torch.cat(
            [encoder_decoder.reconstruct.weight.data / len(other_instances) for encoder_decoder in other_instances],
            dim=1,
        )

        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (X - ica.means_) @ ica.components_.T
        x_scaled = self.scale(x)
        x_projected = self.project(x_scaled)
        return x_projected

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)

    @property
    def is_differentiable(self):
        return True

    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


class NMFEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for NMF encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    other_instances
        Other instances to aggregate. If provided, A is not used.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """

    def __init__(
        self,
        A: torch.tensor = None,
        n_concepts: Optional[int] = None,
        n_features: Optional[int] = None,
        other_instances: Optional[List["NMFEncoderDecoder"]] = None,
        random_state: int = 0,
        save_path: Optional[str] = None,
        force_recompute: bool = False,
    ):
        super().__init__(
            A=A,
            n_concepts=n_concepts,
            n_features=n_features,
            other_instances=other_instances,
        )

        # initialize method
        self.max_iter = 500
        self.random_state = random_state
        self.save_path = os.path.join(save_path)
        self.torch_save_path = os.path.join(self.save_path, "torch_weights.pt")
        self.sklearn_save_path = os.path.join(self.save_path, "sklearn_components.pkl")

        # initialize NMF layers
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=False)

        # initialize scikit-learn NMF
        self.nmf = NMF(n_components=self.n_concepts, random_state=self.random_state, max_iter=self.max_iter)

        self.get_weights(A=A, other_instances=other_instances, force_recompute=force_recompute)

    def _get_weights_from_loading(self):
        # load weights
        self.load_state_dict(torch.load(self.torch_save_path))
        components = pickle.load(open(self.sklearn_save_path, "rb"))

        # set NMF components
        self.nmf.components_ = components
        self.nmf.n_features_in_ = self.n_features

    def _get_weights_from_decomposing(self, A: torch.Tensor):
        # input matrix should be positive
        assert (A >= 0).all(), "NMF only works with non-negative data."

        A = A.cpu().numpy()
        self.nmf.fit(A)

        # set decode weights
        # Xt @ nmf.components_
        self.reconstruct.weight.data = torch.tensor(self.nmf.components_.T)

        if self.save_path is not None:
            torch.save(self.state_dict(), self.torch_save_path)
            pickle.dump(self.nmf.components_, open(self.sklearn_save_path, "wb"))

    def _get_weights_from_aggregating(self, other_instances: List["NMFEncoderDecoder"]):
        """
        Create a new NMFEncoderDecoder by aggregating a list of NMFEncoderDecoder.
        The number of concepts is the sum of the number of concepts of the NMFEncoderDecoder.
        The components are concatenated.

        Parameters
        ----------
        other_instances
            The list of NMFEncoderDecoder to aggregate.

        Returns
        -------
        NMFEncoderDecoder
            The aggregated NMFEncoderDecoder.
        """
        # concatenate decoder weights
        self.reconstruct.weight.data = torch.cat(
            [encoder_decoder.reconstruct.weight.data / len(other_instances) for encoder_decoder in other_instances],
            dim=1,
        )

        # set NMF components
        self.nmf.components_ = self.reconstruct.weight.data.T.cpu().numpy()
        self.nmf.n_features_in_ = self.n_features

        # save aggregated weights
        if self.save_path is not None:
            torch.save(self.state_dict(), self.torch_save_path)
            pickle.dump(self.nmf.components_, open(self.sklearn_save_path, "wb"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        concepts = self.nmf.transform(x.cpu().numpy())
        return torch.tensor(concepts).to(x.device)

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)

    @property
    def is_differentiable(self):
        return False

    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


class PCAEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for PCA encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    other_instances
        Other instances to aggregate. If provided, A is not used.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """

    def __init__(
        self,
        A: torch.tensor = None,
        n_concepts: Optional[int] = None,
        n_features: Optional[int] = None,
        other_instances: Optional[List["PCAEncoderDecoder"]] = None,
        random_state: int = 0,
        save_path: Optional[str] = None,
        force_recompute: bool = False,
    ):
        super().__init__(
            A=A,
            n_concepts=n_concepts,
            n_features=n_features,
            other_instances=other_instances,
        )

        # initialize method
        self.save_path = os.path.join(save_path, "weights.pt")
        self.random_state = random_state

        # initialize PCA layers
        self.scale = nn.Linear(self.n_features, self.n_features, bias=True)
        self.project = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=True)

        self.get_weights(A=A, other_instances=other_instances, force_recompute=force_recompute)

    def _get_weights_from_loading(self):
        self.load_state_dict(torch.load(self.save_path))

    def _get_weights_from_decomposing(self, A: torch.Tensor):
        A = A.numpy()
        pca = PCA(n_components=self.n_concepts, random_state=self.random_state)
        pca.fit(A)

        # set weights
        # set encode scaling
        # X - pca.mean_
        self.scale.weight.data = torch.eye(self.n_features)
        self.scale.bias.data = -torch.tensor(pca.mean_)

        # set encode projection
        # X_scaled @ pca.components_.T
        self.project.weight.data = torch.tensor(pca.components_)

        # set decode
        # X @ pca.components_ + pca.mean_
        self.reconstruct.weight.data = torch.tensor(pca.components_.T)
        self.reconstruct.bias.data = torch.tensor(pca.mean_)

        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def _get_weights_from_aggregating(self, other_instances: List["PCAEncoderDecoder"]):
        """
        Create a new PCAEncoderDecoder by aggregating a list of PCAEncoderDecoder.
        The number of concepts is the sum of the number of concepts of the PCAEncoderDecoder.
        The components are concatenated.

        Parameters
        ----------
        other_instances
            The list of PCAEncoderDecoder to aggregate.

        Returns
        -------
        PCAEncoderDecoder
            The aggregated PCAEncoderDecoder.
        """
        # inherit the constant attributes from the first element of the list
        self.scale.weight.data = other_instances[0].scale.weight.data
        means = torch.cat([encoder_decoder.scale.weight.data for encoder_decoder in other_instances], dim=1).mean(
            dim=1
        )
        self.scale.bias.data = means
        self.reconstruct.bias.data = -means

        # concatenate encoder weights
        self.project.weight.data = torch.cat(
            [encoder_decoder.project.weight.data for encoder_decoder in other_instances], dim=0
        )

        # concatenate decoder weights
        self.reconstruct.weight.data = torch.cat(
            [encoder_decoder.reconstruct.weight.data / len(other_instances) for encoder_decoder in other_instances],
            dim=1,
        )

        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (X - pca.mean_) @ pca.components_.T
        x_scaled = self.scale(x)
        return self.project(x_scaled)

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)

    @property
    def is_differentiable(self):
        return True

    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


class SVDEncoderDecoder(ConceptEncoderDecoder):
    """
    Class for SVD encoding and decoding.
    It decompose the embeddings A into the concept space at initialization.
    Then it can encode and decode embeddings in and from the concept space.

    The concept space is given by U @ Sigma.

    Parameters
    ----------
    A
        The matrix (embeddings) to decompose, of shape (n_samples, n_features).
    n_concepts
        The number of concepts, the dimension of the concept space.
    n_features
        The number of features of the embeddings. If None, it is the number of features of A.
    other_instances
        Other instances to aggregate. If provided, A is not used.
    random_state
        The random state.
    save_path
        The path to save the weights. If a file exist at this location, the weights are loaded.
    force_recompute
        If True, the weights are recomputed even if a file exist at save_path.
    """

    def __init__(
        self,
        A: torch.tensor = None,
        n_concepts: Optional[int] = None,
        n_features: Optional[int] = None,
        other_instances: Optional[List["SVDEncoderDecoder"]] = None,
        random_state: int = 0,
        save_path: Optional[str] = None,
        force_recompute: bool = False,
    ):
        super().__init__(
            A=A,
            n_concepts=n_concepts,
            n_features=n_features,
            other_instances=other_instances,
        )

        # initialize method
        self.save_path = os.path.join(save_path, "weights.pt")
        self.random_state = random_state

        # initialize SVD layers
        self.project = nn.Linear(self.n_features, self.n_concepts, bias=False)
        self.reconstruct = nn.Linear(self.n_concepts, self.n_features, bias=False)

        self.get_weights(A=A, other_instances=other_instances, force_recompute=force_recompute)

    def _get_weights_from_loading(self):
        self.load_state_dict(torch.load(self.save_path))

    def _get_weights_from_decomposing(self, A: torch.Tensor):
        A = A.numpy()
        svd = TruncatedSVD(n_components=self.n_concepts, random_state=self.random_state)
        svd.fit(A)

        # set weights
        # set encode
        # X @ svd.components_.T
        self.project.weight.data = torch.tensor(svd.components_)

        # set decode
        # X @ svd.components_
        self.reconstruct.weight.data = torch.tensor(svd.components_.T)

        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def _get_weights_from_aggregating(self, other_instances: List["SVDEncoderDecoder"]):
        """
        Create a new SVDEncoderDecoder by aggregating a list of SVDEncoderDecoder.
        The number of concepts is the sum of the number of concepts of the SVDEncoderDecoder.
        The components are concatenated.

        Parameters
        ----------
        other_instances
            The list of SVDEncoderDecoder to aggregate.

        Returns
        -------
        SVDEncoderDecoder
            The aggregated SVDEncoderDecoder.
        """
        # concatenate encoder weights
        self.project.weight.data = torch.cat(
            [encoder_decoder.project.weight.data for encoder_decoder in other_instances], dim=0
        )

        # concatenate decoder weights
        self.reconstruct.weight.data = torch.cat(
            [encoder_decoder.reconstruct.weight.data / len(other_instances) for encoder_decoder in other_instances],
            dim=1,
        )

        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

    def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x_encoded)

    @property
    def is_differentiable(self):
        return True

    @property
    def concepts_base(self):
        # (n_concepts, n_features)
        return self.reconstruct.weight.data.T


# class KMeansEncoderDecoder(ConceptEncoderDecoder):
#     """
#     Class for KMeans concepts discovery.
#     It finds concepts through KMeans clustering. Concepts are the centroids of the clusters.
#     Hence concepts encoding is a similarity to concepts.
#     But concepts cannot be decoded.

#     Parameters
#     ----------
#     A
#         The matrix (embeddings) to decompose, of shape (n_samples, n_features).
#     n_concepts
#         The number of concepts, the dimension of the concept space.
#     random_state
#         The random state.
#     stock_concepts
#         If True, the concepts obtained from A are stored in the object. Only at initialization.
#     """
#     def __init__(self, A: torch.tensor,
#                  n_concepts: Optional[int] = None,
#                  random_state: int = 0,
#                  stock_concepts: bool = False):
#         # initialize method
#         super().__init__(self, n_concepts)
#         self.is_differentiable = False
#         # initialize method
#         super().__init__(self, n_concepts)

#         # compute KMeans
#         A = A.numpy()
#         self.kmeans = KMeans(n_clusters=self.n_concepts, random_state=random_state)
#         if stock_concepts:
#             self.concepts = 1 / (1 + self.kmeans.fit_transform(A))
#         else:
#             self.kmeans.fit(A)

#         # save centroids: shape (n_concepts, n_features)
#         self.centroids = torch.tensor(self.kmeans.cluster_centers_)

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         centroid_distances = self.kmeans.transform(x.numpy())
#         centroid_similarities = 1 / (1 + centroid_distances)
#         return torch.tensor(centroid_similarities).type(torch.float32)

#     def decode(self, x_encoded: torch.Tensor) -> torch.Tensor:
#         # warning: decoding is not possible with KMeans
#         Warning("Decoding is not possible with KMeans, the result will not be meaningful.")
#         return (x_encoded @ self.centroids) / x_encoded.sum(axis=1, keepdim=True)  # TODO: remove if the other work

# @property
# @abstractmethod
# def is_differentiable(self):
#     return self.is_differentiable

# The following code is too long, I did not find a way to make it work
# x_encoded = x_encoded.numpy()
# n_samples, n_features = x_encoded.shape[0], self.centroids.shape[1]

# # objective function: Minimize the sum of squared differences between actual and computed distances
# def objective(x):
#     x = x.reshape((n_samples, n_features))
#     diff = np.expand_dims(self.centroids, axis=0) - np.expand_dims(x, axis=1)
#     dist = np.sqrt(np.sum(diff**2, axis=2))
#     return np.sum((dist - (1 / x_encoded - 1))**2)

# # initial guess for the new point's coordinates
# x0 = (x_encoded @ self.centroids) / np.sum(x_encoded, axis=1, keepdims=True)

# # print("x_encoded", x_encoded.shape)  # TODO: remove
# # print("centroids", self.centroids.shape)
# # print("x_encoded @ centroids", (x_encoded @ self.centroids).shape)
# # print("sum", np.sum(x_encoded, axis=1, keepdims=True).shape)
# # print("x0", x0.shape)
# # print("centroids - x0", (np.expand_dims(self.centroids, axis=0) - np.expand_dims(x0, axis=1)).shape)
# # print("dist", np.sqrt(np.sum((np.expand_dims(self.centroids, axis=0) - np.expand_dims(x0, axis=1))**2, axis=2)).shape)

# x0 = x0.flatten()

# # Use 'minimize' to find the best fit point
# result = minimize(objective, x0, method='BFGS')
# return result.x
