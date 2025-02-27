"""
Generic type annotations for Interpreto
"""

from __future__ import annotations

from typing import Any

import torch

TokenEmbedding = Any
Activation = Any
ModelInput = Any
TensorBaseline = torch.Tensor | float | int | None
