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

import re

from torch import nn


class ModelPathError(ValueError):
    """Raised to signal a problem with model path definitions."""


def validate_path(module: nn.Module, path: str) -> None:
    """Validates that a specified model path is present in the loaded model."""
    # Split the given path into components
    components = re.findall(r"([^\.]+)", path)
    current = module
    current_path = ""
    for name in components:
        try:
            current = getattr(current, name)
            if not current_path:
                current_path = name
            else:
                current_path += f".{name}"
        except AttributeError as ex:
            # Get available submodules for more informative error
            try:
                available_submodules = list(current._modules.keys())
            except Exception:
                available_submodules = []
            raise ModelPathError(
                f"The provided splitting point '{path}' is not valid.\n"
                f"Module {current_path} does not have submodule '{name}'.\n"
                f"Available submodule names: {', '.join(available_submodules)}.\n"
                "Use 'model.[...] to define a split based on the structure of your model, or pass an integer "
                "corresponding to the selected layer to use the output of that layer as splitting point."
            ) from ex


def walk_modules(module: nn.Module, prefix=""):
    """Recursively walk through the model yielding all model paths.

    Args:
        module (torch.nn.Module): The module to walk through
        prefix (str): Accumulated path prefix

    Yields:
        str: Full path to each module
    """
    for name, child in module._modules.items():
        current_path = f"{prefix}.{name}" if prefix else name
        if child is not None and len(list(child.children())) > 0:
            yield from walk_modules(child, current_path)
        yield current_path


def get_path_idx(split: str, model_paths: list[str]) -> int:
    """Match a model path to its index in the model according to the order of forward pass completion.

    Args:
        split (str): Split path to match

    Returns:
        int: Index in model_paths, or raises an error if no match
    """
    if split not in model_paths:
        raise ModelPathError(f"Split '{split}' not found in available model modules.")
    return model_paths.index(split)


def sort_paths(module: nn.Module, splits: str | list[str]) -> list[str]:
    """Order model paths according to their actual occurrence in the model's forward pass."""
    splits = splits if isinstance(splits, list) else [splits]
    model_paths = list(walk_modules(module))
    return sorted(splits, key=lambda split: get_path_idx(split, model_paths=model_paths))
