"""
Concept Bottleneck Explainer based on Overcomplete concept-encoder-decoder framework.
"""

from __future__ import annotations

from typing import NamedTuple

from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae


class OvercompleteMethods(NamedTuple):
    """
    Overcomplete dictionary learning methods.
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


def OvercompleteCBE(model: ...):
    """
    Implementation of a concept explainer based on the Overcomplete framework.
    """

    def fit(self, inputs, split):
        """
        Method for explaining a concept.

        :param inputs: The inputs to be explained.
        :return: the concept encoder and the concept decoder.
        """
        ...

    def concept_output_attribution(self, concepts, attribution_method): ...
