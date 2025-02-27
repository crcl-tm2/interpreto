"""
Generic type annotations for Interpreto
"""

from __future__ import annotations

from typing import Any, Union

import torch

TokenEmbedding = Any
Activation = Any
ModelInput = Any
TensorBaseline = Union[torch.Tensor, float, int, type(None)]  # noqa: UP007
