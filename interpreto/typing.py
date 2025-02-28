"""
Generic type annotations for Interpreto
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

TokenEmbedding = torch.Tensor
LatentActivation = torch.Tensor
ConceptsActivation = torch.Tensor
ModelInput = Any
TensorBaseline = Optional[Union[torch.Tensor, float, int]]  # noqa: UP007
