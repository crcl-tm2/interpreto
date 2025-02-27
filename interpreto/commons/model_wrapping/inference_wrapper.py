from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch


class ClassificationInferenceWrapperPlaceholder:  # TODO: remove this class
    def __init__(self, model: torch.nn.Module, batch_size: int, device: torch.device | None = None):
        self.model = model
        assert batch_size > 0
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cpu")

    def to(self, device: torch.device):
        self.model.to(device)

    def cpu(self):
        self.model.cpu()

    def cuda(self):
        self.model.cuda()

    def flatten_unflatten(func: Callable) -> Callable:
        """
        A decorator that flattens multiple batch dimensions before calling the function
        and unflattens the output back to the original shape.

        It introduces a 'flatten' argument to control this behavior.

        Args:
            func (Callable): The function to wrap.

        Returns:
            Callable: The wrapped function.
        """

        @functools.wraps(func)
        def wrapper(self, inputs: torch.Tensor, *args: Any, flatten: bool = False, **kwargs: Any) -> torch.Tensor:
            """
            Wrapper that flattens and unflattens the inputs tensor based on the 'flatten' flag.

            Args:
                inputs (torch.Tensor): Inputs tensor of shape (n, p, ...).
                flatten (bool): Whether to flatten before and unflatten after.

            Returns:
                torch.Tensor: Processed tensor with restored shape if flatten=True.
            """
            if not isinstance(inputs, torch.Tensor):
                raise TypeError("Expected 'inputs' to be a PyTorch tensor.")

            orig_shape = inputs.shape  # Store original shape

            # Flatten if requested
            if flatten:
                inputs = inputs.flatten(start_dim=0, end_dim=1)  # Shape: (n*p, ...)

            # Call the original function
            outputs = func(self, inputs, *args, **kwargs)

            # Unflatten if needed
            if flatten and isinstance(outputs, torch.Tensor):
                outputs = outputs.unflatten(dim=0, sizes=orig_shape[:2])  # Restore shape: (n, p, output_dim)

            return outputs

        return wrapper

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        predicted_class = logits.argmax(dim=-1)
        selected_logits = torch.gather(logits, dim=-1, index=predicted_class.unsqueeze(-1)).squeeze(-1)
        return selected_logits

    def gradients(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clone().detach().requires_grad_(True)  # TODO: verify the clone, not sure useful
        scores = self.inference(inputs)
        scores.backward(torch.ones_like(scores))  # Allow multiple sample dimensions (n, p)
        return inputs.grad

    @flatten_unflatten
    def batch_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        scores = []
        for i in range(0, inputs.shape[0], self.batch_size):
            batch = inputs[i : i + self.batch_size].to(self.device)
            batch_scores = self.inference(batch).cpu()
            scores.append(batch_scores)
        return torch.cat(scores, dim=0)

    @flatten_unflatten
    def batch_gradients(self, inputs: torch.Tensor) -> torch.Tensor:
        gradients = []
        for i in range(0, inputs.shape[0], self.batch_size):
            batch = inputs[i : i + self.batch_size].to(self.device)
            batch_gradients = self.gradients(batch).detach().cpu()
            gradients.append(batch_gradients)
        return torch.cat(gradients, dim=0)
