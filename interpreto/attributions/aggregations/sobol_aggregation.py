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

import numpy as np
import torch

from interpreto.attributions.aggregations.base import Aggregator


class SobolAggregator(Aggregator):
    """
    Aggregates Sobol indices from model outputs.
    """

    @staticmethod
    def aggregate(f_orig, dict_f_hybrid):
        """
        Compute the Sobol indices from the model outputs on the origin perturbations (f_orig)
        and the token-specific perturbations (dict_f_hybrid).

        Returns a dictionary mapping the token index to its Sobol attribution index.
        """
        # Convert to numpy if necessary.
        if torch.is_tensor(f_orig):
            f_orig = f_orig.cpu().detach().numpy()
        var_f = np.var(f_orig)
        # To avoid division by zero.
        if var_f == 0:
            var_f = 1e-6
        S = {}
        for token_idx, f_hybrid in dict_f_hybrid.items():
            if torch.is_tensor(f_hybrid):
                f_hybrid = f_hybrid.cpu().detach().numpy()
            delta = f_orig - f_hybrid
            S[token_idx] = np.mean(delta**2) / var_f
        return S
