"""
Generic type annotations for Interpreto
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import torch

TokenEmbedding = Any
Activation = Any
ModelInput = str | Iterable[str] | Mapping[str, torch.Tensor] | torch.Tensor
TensorBaseline = Optional[Union[torch.Tensor, float, int]]  # noqa: UP007
