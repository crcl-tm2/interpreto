"""
Generic type annotations for Interpreto
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import torch

TokenEmbedding = Any
Activation = Any
ModelInputs = str | Mapping[str, torch.Tensor] | Iterable[str] |Iterable[Mapping[str, torch.Tensor]]
TensorBaseline = Optional[Union[torch.Tensor, float, int]]  # noqa: UP007
