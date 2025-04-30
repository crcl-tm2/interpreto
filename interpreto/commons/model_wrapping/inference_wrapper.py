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
Basic inference wrapper for explaining models.

This module provides a base class for inference wrappers that can be used to
perform inference on various models. The InferenceWrapper class is designed to
handle device management, embedding inputs, and batching of inputs for efficient
processing. The class is designed to be subclassed for specific model types and tasks.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable, MutableMapping
from functools import singledispatchmethod
from typing import Any, overload

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from interpreto.typing import ModelInputs, TensorMapping


# TODO : move that somewhere else
def concat_and_pad(
    *tensors: torch.Tensor | None,
    pad_left: bool,
    dim: int = 0,
    pad_value: int = 0,
    pad_dims: Iterable[int] | None = None,
) -> torch.Tensor:
    """
    Concatenate and pad tensors to the maximum length of each dimension.

    Args:
        *tensors (torch.Tensor | None): tensors to concatenate (can be of different shapes but must have the same number of dimensions). Can be None.
        pad_left (bool): if True, padding is done on the left side of the tensor, otherwise on the right side.
        dim (int, optional): Dimension along which the tensors will be concatenated. Defaults to 0.
        pad_value (int, optional): Value used to pad the tensors. Defaults to 0.
        pad_dims (Iterable[int] | None, optional): Dimensions to pad. Defaults to None.

    Returns:
        torch.Tensor: result of the concatenation

    Raises:
        ValueError: if the tensors have different number of dimensions.
        TypeError: If the `tensors` argument is not a valid sequence of tensors or if
            `pad_dims` contains invalid dimensions.

    Example:
        >>> t1 = torch.randn(2, 3, 4)
        >>> t2 = torch.randn(3, 2, 5)
        >>> t3 = torch.randn(1, 6, 1)
        >>> result = concat_and_pad(t1, t2, t3, pad_left=True, dim=0, pad_value=-1, pad_dims=[1, 2])
        >>> print(result.shape)
        torch.Size([6, 6, 5])  # After padding and concatenation along the first dimension
    """
    _tensors = [a for a in tensors if a is not None and a.numel()]
    if not _tensors:
        raise ValueError("No tensors provided for concatenation.")
    if any(t.dim() != _tensors[0].dim() for t in _tensors[1:]):
        raise ValueError("All tensors must have the same number of dimensions.")
    tensors_dim = _tensors[0].dim()
    pad_dims = pad_dims or []
    max_length_per_dim = [max(t.shape[d] for t in _tensors) for d in pad_dims]

    padded_tensors: list[torch.Tensor] = []
    for t in _tensors:
        pad = [0, 0] * tensors_dim
        for pad_dim, pad_length in zip(pad_dims, max_length_per_dim, strict=True):
            # update padding indication to pad the right dimension
            pad_index = -2 * (pad_dim % tensors_dim) - 1 - pad_left
            pad[pad_index] = pad_length - t.shape[pad_dim]
        # pad the tensor
        padded_tensors.append(torch.nn.functional.pad(t, pad, value=pad_value))
    # return the concatenation of all tensors
    return torch.cat(padded_tensors, dim=dim)


class InferenceWrapper:
    """
    Base class for inference wrapper objects.
    This class is designed to wrap a model and provide a consistent interface for
    performing inference on the model's inputs. It handles device management,
    embedding inputs, and batching of inputs for efficient processing.
    The class is designed to be subclassed for specific model types and tasks.

    Attributes:
        model (PreTrainedModel): The model to be wrapped.
        batch_size (int): The maximum batch size for processing inputs.
        device (torch.device | None): The device on which the model is loaded.
    """

    # static attribute to indicate whether to pad on the left or right side
    # this is a class attribute and should be set in subclasses
    PAD_LEFT = True

    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int,
        device: torch.device | None = None,
    ):
        self.model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.batch_size = batch_size

        # Pad token id should be set by the explainer
        self.pad_token_id = None

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: The device on which the model is loaded.
        """
        return self.model.device

    @device.setter
    def device(self, device: torch.device):
        """
        Sets the device on which the model is loaded.

        Args:
            device (torch.device): wanted device (e.g., "cpu" or "cuda").
        """
        self.model.to(device)

    def to(self, device: torch.device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The device to which the model should be moved.
        """
        self.device = device

    def cpu(self):
        """
        Move the model to the CPU.
        """
        self.device = torch.device("cpu")

    def cuda(self):
        """
        Move the model to the GPU.
        """
        self.device = torch.device("cuda")

    def embed(self, model_inputs: TensorMapping) -> TensorMapping:
        """
        Embed the inputs using the model's input embeddings.

        Args:
            model_inputs (TensorMapping): input mapping containing either "input_ids" or "inputs_embeds".

        Raises:
            ValueError: If neither "input_ids" nor "inputs_embeds" are present in the input mapping.

        Returns:
            TensorMapping: The input mapping with "inputs_embeds" added.
        """
        # If input embeds are already present, return the unmodified model inputs
        if "inputs_embeds" in model_inputs:
            return model_inputs
        # If input ids are present, get the embeddings and add them to the model inputs
        if "input_ids" in model_inputs:
            base_shape = model_inputs["input_ids"].shape
            input_ids = model_inputs["input_ids"].flatten(0, -2).to(self.device)
            flatten_embeds = self.model.get_input_embeddings()(input_ids)
            model_inputs["inputs_embeds"] = flatten_embeds.view(*base_shape, flatten_embeds.shape[-1])
            return model_inputs
        # If neither input ids nor input embeds are present, raise an error
        raise ValueError("model_inputs should contain either 'input_ids' or 'inputs_embeds'")

    def call_model(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor | None) -> BaseModelOutput:
        """
        Perform a call to the wrapped model with the given input embeddings and attention mask.

        Args:
            input_embeds (torch.Tensor): embedded inputs
            attention_mask (torch.Tensor): attention mask

        Returns:
            ModelOutput: The output of the model.

        Note:
            If the batch size of the input embeddings exceeds the wrapper's batch size, a warning is issued.
        """
        # Check that batch size of input_embeds is not greater than the wrapper's batch size
        if input_embeds.shape[0] > self.batch_size:
            warnings.warn(
                f"Batch size of {input_embeds.shape[0]} is greater than the wrapper's batch size of {self.batch_size}. "
                f"Consider adjust the batch size or the wrapper of split your data.",
                stacklevel=1,
            )
        # send input to device
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # Call wrapped model
        return self.model(inputs_embeds=input_embeds.to(self.device), attention_mask=attention_mask)

    @overload
    def get_logits(self, model_inputs: TensorMapping) -> torch.Tensor: ...

    @overload
    def get_logits(self, model_inputs: Iterable[TensorMapping]) -> Generator[torch.Tensor, None, str]: ...

    @singledispatchmethod
    def get_logits(self, model_inputs: ModelInputs) -> torch.Tensor | Generator[torch.Tensor, None, str]:
        """
        Get the logits from the model for the given inputs.

        This method propose two different treatments of the inputs:
        If the input is a mapping, it will be processed as a single input and given directly to the model.
        The method will return the logits of the model as a torch.Tensor.

        If the input is an iterable of mappings, it will be processed as a batch of inputs.
        The method will yield the logits of the model for each input as a torch.Tensor.

        Args:
            model_inputs (Any): input mappings to be passed to the model or iterable of input mappings.

        Raises:
            NotImplementedError: If the input type is not supported.

        Returns:
            torch.Tensor | Generator[torch.Tensor, None, None]: logits associated to the input mappings.

        Example:
            Single input given as a mapping
                >>> model_inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
                >>> logits = wrapper.get_logits(model_inputs)
                >>> print(logits.shape)

            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> logits = wrapper.get_logits(model_inputs)
                >>> for logit in logits:
                ...     print(logit.shape)

        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}"
        )

    @get_logits.register(MutableMapping)  # type: ignore
    def _get_logits_from_mapping(self, model_inputs: TensorMapping) -> torch.Tensor:
        """
        Get the logits from the model for the given inputs.
        registered for MutableMapping type.

        Args:
            model_inputs (TensorMapping): input mapping containing either "input_ids" or "inputs_embeds".

        Returns:
            torch.Tensor: logits associated to the input mapping.
        """
        # if emebeddings has not been calculated yet, embed the inputs
        model_inputs = self.embed(model_inputs)
        # depending on the number of dimensions of the input
        match model_inputs["inputs_embeds"].dim():
            case 2:  # (sequence_length, embedding_size)
                return self.call_model(**model_inputs).logits
            case 3:  # (batch_size, sequence_length, embedding_size)
                # If a batch dimension is given, split the inputs into chunks of batch_size
                embeds_chunks = model_inputs["inputs_embeds"].split(self.batch_size)
                mask_chunks = model_inputs["attention_mask"].split(self.batch_size)

                # call the model on each chunk and concatenate the results
                return torch.cat(
                    [
                        self.call_model(embeds_chunk, mask_chunk).logits
                        for embeds_chunk, mask_chunk in zip(embeds_chunks, mask_chunks, strict=False)
                    ],
                )
            case _:  # (..., sequence_length, embedding_size) e.g. (batch_size, n_perturbations, sequence_length, embedding_size)
                # flatten the first dimension to a single batch dimension
                # then call the model on the flattened inputs and reshape the result to the original batch structure
                flat_model_inputs = {
                    "inputs_embeds": model_inputs["inputs_embeds"].flatten(0, -3),
                    "attention_mask": model_inputs["attention_mask"].flatten(0, -2),
                }
                prediction = self._get_logits_from_mapping(flat_model_inputs)
                return prediction.view(*model_inputs["inputs_embeds"].shape[:-2], -1)

    @get_logits.register(Iterable)  # type: ignore
    def _get_logits_from_iterable(self, model_inputs: Iterable[TensorMapping]) -> Generator[torch.Tensor, None, str]:
        """
        Get the logits from the model for the given inputs.
        registered for Iterable type.
        Args:
            model_inputs (Iterable[TensorMapping]): Iterable of input mappings containing either "input_ids" or "inputs_embeds".
        Yields:
            torch.Tensor: logits associated to the input mappings.
        """
        for model_input in model_inputs:
            yield self._get_logits_from_mapping(model_input)

    def get_logits_from_iterable_error(
        self, model_inputs: Iterable[TensorMapping]
    ) -> Generator[torch.Tensor, None, str]:
        """
        Get the logits from the model for the given inputs.
        registered for Iterable type.

        Args:
            model_inputs (Iterable[TensorMapping]): Iterable of input mappings containing either "input_ids" or "inputs_embeds".

        Yields:
            torch.Tensor: logits associated to the input mappings.
        """
        # create an iterator from the input iterable
        model_inputs = iter(model_inputs)

        # If no pad token id has been given
        if self.pad_token_id is None:
            # raise ValueError(
            #     "Asking to pad but the tokenizer does not have a padding token. Please select a token to use as pad_token (tokenizer.pad_token = tokenizer.eos_token e.g.) or add a new pad token via tokenizer.add_special_tokens({'pad_token': '[PAD]'})"
            # )
            raise ValueError(
                "Padding token is not set in the inference wrapper. Please assign it explicitly by setting: inference_wrapper.pad_token_id = tokenizer.pad_token_id"
            )

        result_buffer = torch.zeros(0)
        result_indexes: list[int] = []

        batch: torch.Tensor | None = None
        batch_mask: torch.Tensor | None = None

        input_buffer = torch.zeros(0)
        mask_buffer = torch.zeros(0)

        last_item = False

        # Generation loop
        while True:
            # check if the ouput buffer contains enough data to correspond to the next element
            if result_buffer.numel() and result_indexes and len(result_buffer) >= result_indexes[0]:
                # pop the first index from the result indexes
                index = result_indexes.pop(0)
                # yield the associated logits
                yield result_buffer[:index]
                # remove the yielded logits from the result buffer
                result_buffer = result_buffer[index:]
                # if there is no element left in the input stream and the result buffer is empty, break the loop
                if last_item and not result_indexes:
                    break
                continue
            # check if the batch of inputs is large enough to be processed (or if the last item is reached)
            if batch is not None and (last_item or len(batch) == self.batch_size):
                # Call the model
                logits = self.call_model(batch, batch_mask).logits
                # Concatenate the results to the output buffer

                ##################### FIXME #####################
                # The .detach().clone() if used to avoid memory issues provoked by the bad usage of the result_buffer
                # This will block the gradient calculation on the yielded logits
                # Gradient calculation currently only call _get_logits_from_mapping register for the jacobian calculation
                # This code works but should be improved in the future
                ###############################################
                result_buffer = concat_and_pad(result_buffer, logits, pad_left=self.PAD_LEFT).detach().clone()
                # update batch and mask
                batch = batch_mask = None
                continue
            # check if the input buffer contains enough data to fill the batch
            if input_buffer.numel():
                # calculate the missing length of the batch
                missing_length = self.batch_size - len(batch if batch is not None else ())
                # fill the batch with the missing data
                batch = concat_and_pad(
                    batch,
                    input_buffer[:missing_length],
                    pad_left=self.PAD_LEFT,
                    dim=0,
                    pad_value=self.pad_token_id,
                    pad_dims=(1,),
                )
                batch_mask = concat_and_pad(
                    batch_mask,
                    mask_buffer[:missing_length],
                    pad_left=self.PAD_LEFT,
                    dim=0,
                    pad_value=0,
                    pad_dims=(-1,),
                )
                # remove the used data from the input buffer
                input_buffer = input_buffer[missing_length:]
                mask_buffer = mask_buffer[missing_length:]
                continue
            # If there is not enough data in the input buffer, get the next item from the input stream
            try:
                # Get next item and ensure embeddings are calculated
                next_item = self.embed(next(model_inputs))
                input_buffer = self._reshape_inputs(next_item["inputs_embeds"], non_batch_dims=2)
                mask_buffer = self._reshape_inputs(next_item["attention_mask"], non_batch_dims=1)
                # update results index list
                result_indexes += [len(input_buffer)]
            # If the input stream is empty
            except StopIteration:
                if last_item:
                    # This should never happen
                    break
                last_item = True
        # Chack that all the buffers are empty
        if any(element.numel() for element in [result_buffer, input_buffer, mask_buffer]):
            warnings.warn(
                "Some data were not well fetched in inference wrapper, please check your code if you made custom method or notify it to the developers",
                stacklevel=2,
            )
            return "Some data were not well fetched in inference wrapper"
        return "All data were well fetched in inference wrapper"

    def _reshape_inputs(self, tensor: torch.Tensor, non_batch_dims: int = 2) -> torch.Tensor:
        """
        reshape inputs to have a single batch dimension.
        """
        # TODO : see if there is a better way to do this
        assert tensor.dim() >= non_batch_dims, "The given tensor have less dimensions than non_batch_dims parameter"
        if tensor.dim() == non_batch_dims:
            return tensor.unsqueeze(0)
        if tensor.dim() == non_batch_dims + 1:
            return tensor
        assert tensor.shape[0] == 1, (
            "When passing a sequence or a generator of inputs to the inference wrapper, please consider giving sequence of perturbations of single elements instead of batches (shape should be (1, n_perturbations, ...))"
        )
        return self._reshape_inputs(tensor[0], non_batch_dims=non_batch_dims)

    @singledispatchmethod
    def get_targeted_logits(
        self, model_inputs: Any, targets: torch.Tensor
    ) -> torch.Tensor | Generator[torch.Tensor, None, None]:
        raise NotImplementedError(
            f"get_targeted_logits not implemented for {self.__class__.__name__}. Implement this method is necessary to use gradient-based methods."
        )

    @overload
    def get_gradients(self, model_inputs: TensorMapping, targets: torch.Tensor) -> torch.Tensor: ...

    @overload
    def get_gradients(
        self, model_inputs: Iterable[TensorMapping], targets: Iterable[torch.Tensor]
    ) -> Iterable[torch.Tensor]: ...

    # @allow_nested_iterables_of(MutableMapping)
    @singledispatchmethod
    def get_gradients(
        self, model_inputs: ModelInputs, targets: torch.Tensor
    ) -> torch.Tensor | Generator[torch.Tensor, None, None]:
        """
        Get the gradients of the logits associated to a given target with respect to the inputs.

        Args:
            model_inputs (Any): input mappings to be passed to the model or iterable of input mappings.
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Raises:
            NotImplementedError: If the input type is not supported.

        Returns:
            torch.Tensor|Generator[torch.Tensor, None, None]: gradients of the logits.

        Example:
            Single input given as a mapping
                >>> model_inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
                >>> targets = torch.tensor([1, 2])
                >>> gradients = wrapper.get_gradients(model_inputs, targets)
                >>> print(gradients)
            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> targets = torch.tensor([[1, 2], [3, 4]])
                >>> gradients = wrapper.get_gradients(model_inputs, targets)
                >>> for grad in gradients:
                ...     print(grad)
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register(MutableMapping)  # type: ignore
    def _get_gradients_from_mapping(self, model_inputs: TensorMapping, targets: torch.Tensor) -> torch.Tensor:
        model_inputs = self.embed(model_inputs)
        inputs_embeds = model_inputs["inputs_embeds"]

        def get_score(inputs_embeds: torch.Tensor):
            return self.get_targeted_logits(
                {"inputs_embeds": inputs_embeds, "attention_mask": model_inputs["attention_mask"]}, targets
            )

        # Compute gradient of the selected logits:
        grad_matrix = torch.autograd.functional.jacobian(get_score, inputs_embeds)  # (n, lt, n, l, d)
        grad_matrix = grad_matrix.abs().mean(dim=-1)  # (n, lt, n, l)  # average over the embedding dimension
        return grad_matrix[torch.arange(grad_matrix.shape[0]), :, torch.arange(grad_matrix.shape[0]), :]

    @get_gradients.register(Iterable)  # type: ignore
    def _get_gradients_from_iterable(
        self, model_inputs: Iterable[TensorMapping], targets: Iterable[torch.Tensor]
    ) -> Iterable[torch.Tensor]:
        for model_input, target in zip(model_inputs, targets, strict=True):
            # check that the model input and target have the same batch size
            result = self._get_gradients_from_mapping(model_input, target)
            yield result

        # yield from (
        #     self.get_gradients(model_input, target) for model_input, target in zip(model_inputs, targets, strict=True)
        # )
