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

from collections.abc import Iterable, MutableMapping
from functools import singledispatchmethod

import torch
from torch.autograd.functional import jacobian

from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper


class GenerationInferenceWrapper(InferenceWrapper):
    PAD_LEFT = True

    @singledispatchmethod
    def get_inputs_to_explain_and_targets(self, model_inputs, **generation_kwargs):
        """
        Abstract method to prepare inputs and targets for explanation in a generation setting.

        This method should be implemented for different input types (e.g., MutableMapping, Iterable).
        It takes inputs to the model and returns a tuple:
            - A full input including the original prompt and the generated continuation.
            - The target token IDs (i.e., the generated part only), used for computing gradients or scores.

        Parameters:
            model_inputs: The model inputs. Can be a single MutableMapping or an Iterable
                of MutableMappings. Each MutableMapping should contain at least "input_ids" or "input_embeds" and "attention_mask"
                (as expected by Hugging Face models), representing one or multiple sequences.
            **generation_kwargs: Optional keyword arguments passed directly to `model.generate()`.
                These control the decoding strategy and can include:
                    - `max_new_tokens` (int): maximum number of new tokens to generate.
                    - `do_sample` (bool): whether to sample (True) or use greedy decoding (False).
                    - `temperature` (float): sampling temperature, higher = more randomness.
                    - `top_k` (int): restrict sampling to the top-k most probable tokens.
                    - `top_p` (float): nucleus sampling probability threshold.
                    - `num_beams` (int): number of beams for beam search decoding.
                    - ... and any other supported generation parameter from Hugging Face.
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_inputs_to_explain_and_targets in class {self.__class__.__name__}"
        )

    @get_inputs_to_explain_and_targets.register(MutableMapping)
    def _(
        self, model_inputs: MutableMapping[str, torch.Tensor], **generation_kwargs
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor]:
        """
        Given a batch of input sequences (as a single MutableMapping), this method generates continuations
        using the model and returns:

            1. A new MutableMapping that contains the full sequences (input + generated tokens) and updated
               attention masks.
            2. A tensor containing only the token IDs of the generated part (targets).

        Input:
            - model_inputs: a MutableMapping (e.g., a dictionary with "input_ids" and "attention_mask")
              corresponding to one or multiple input sequences in a batch.
            - generation_kwargs: additional keyword arguments passed to `model.generate()`
              (e.g., max_new_tokens, do_sample, temperature).

        Process:
            - Calls `model.generate()` to produce full sequences including both input and generated tokens.
            - Extracts the generated part (i.e., the new tokens after the input) based on the original
              attention mask length.
            - Concatenates the original inputs and generated tokens into a new MutableMapping with updated
              attention masks.
            - Returns the full MutableMapping and the target token IDs (i.e., generated part only).

        Returns:
            A tuple of:
                - full_mapping: a dictionary with keys like "input_ids" and "attention_mask" that
                  now include both input and generated tokens.
                - targets_ids: a tensor containing only the generated token IDs.
        """
        full_ids = self.model.generate(**model_inputs, **generation_kwargs)
        original_length = model_inputs["attention_mask"].shape[-1]
        targets_ids = full_ids[..., original_length:]
        full_attention_mask = torch.cat([model_inputs["attention_mask"], torch.ones_like(targets_ids)], dim=-1)
        full_mapping = {"input_ids": full_ids, "attention_mask": full_attention_mask}
        return full_mapping, targets_ids

    @get_inputs_to_explain_and_targets.register(Iterable)
    def _(
        self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], **generation_kwargs
    ) -> tuple[MutableMapping[str, torch.Tensor], torch.Tensor]:
        """
        Applies get_inputs_to_explain_and_targets to each MutableMapping in an iterable (e.g., a list of batched inputs).

        Returns:
            A list of tuples (full_mapping, targets_ids) for each element in the input iterable.
        """
        l_full_mappings, l_targets_ids = [], []
        for model_input in model_inputs:
            full_mappings, targets_ids = self.get_inputs_to_explain_and_targets(model_input, **generation_kwargs)
            l_full_mappings.append(full_mappings)
            l_targets_ids.append(targets_ids)
        return l_full_mappings, l_targets_ids

    @singledispatchmethod
    def get_targeted_logits(self, model_inputs, targets):
        """
        Abstract method to retrieve the logits associated with the target tokens.
        Must be implemented per input type (e.g., MutableMapping, Iterable).
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targeted_logits in class {self.__class__.__name__}"
        )

    @get_targeted_logits.register(MutableMapping)
    def _(self, model_inputs: MutableMapping[str, torch.Tensor], targets: torch.Tensor):
        """
        Retrieves the logits corresponding to the target token IDs for a single MutableMapping
        (i.e., a batch of inputs stored as a dictionary of tensors).

        Input:
            - model_inputs: a single MutableMapping (typically a dictionary) containing the batched inputs
              for the model. These inputs correspond to the full sequence for each example in the batch,
              including both the original prompt and the generated tokens (i.e., input + generated/target).
            - targets: a tensor of shape (batch_size, target_length) containing the token IDs of the
              generated part (the continuation) for each input in the batch.

        Steps:
            - The last token of each input sequence is removed to simulate next-token prediction.
            - The truncated inputs are embedded and passed through the model to obtain logits for each position
              and vocabulary token.
            - We slice the output logits to keep only the positions that correspond to the generated tokens.
            - Finally, we filter the vocabulary dimension by gathering only the logits that correspond to the
              actual target token IDs (instead of keeping the logits for all vocabulary items).

        Returns:
            - A tensor of shape (batch_size, target_length) containing the predicted logits for the target
              tokens in each sequence of the batch.
        """
        # remove last target token from the model inputs
        # to avoid using the last token in the generation process
        model_inputs = {
            key: value[..., :-1, :] if key == "inputs_embeds" else value[..., :-1]
            for key, value in model_inputs.items()
        }
        model_inputs = self.embed(model_inputs)

        # Get complete logits regardless of the input's shape.
        logits = self.get_logits(model_inputs)  # (l-1, v) | (n, l-1, v) | (n, p, l-1, v)

        target_length = targets.shape[-1]  # lt < l

        # assume the sequence dimension is the second-to-last.
        target_logits = logits[..., -target_length:, :]
        if targets.shape != target_logits.shape[:-1]:
            raise ValueError(
                "target logits shape without the vocabulary dimension must match the target inputs ids shape."
                f"Got {target_logits.shape[:-1]} and {targets.shape}."
            )

        # For a batch case, unsqueeze the targets so that they match the logits shape.
        selected_logits = target_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        return selected_logits

    # @get_targeted_logits.register(Iterable)
    # def _(self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], targets: Iterable[torch.Tensor]):
    #     # Get the logits for each model input and target pair.
    #     if len(model_inputs) != len(targets):
    #         raise ValueError(
    #             f"model_inputs and targets must have the same length. Got {len(model_inputs)} and {len(targets)}."
    #         )
    #     selected_logits = []
    #     for model_input, target in zip(model_inputs, targets, strict=True):
    #         p = model_input["attention_mask"].shape[0]
    #         p_targets = target.unsqueeze(0).repeat(p, 1)
    #         selected_logits.append(self.get_targeted_logits(model_input, p_targets))
    #     return torch.stack(selected_logits)

    @get_targeted_logits.register(Iterable)
    def _(self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], targets: Iterable[torch.Tensor]):
        """
        Retrieves logits for each pair of model input and target in an iterable.
        """
        all_logits = self.get_logits(model_inputs)
        for logits, target in zip(all_logits, targets, strict=True):
            target_length = target.shape[-1]
            targeted_logits = logits[..., -target_length:, :]
            selected_logits = targeted_logits.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            yield selected_logits

    @singledispatchmethod
    def get_gradients(self, model_inputs, targets):
        """
        Abstract method to compute gradients of target logits w.r.t. input embeddings.
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register(MutableMapping)
    def _(self, model_inputs: MutableMapping[str, torch.Tensor], targets: torch.Tensor):
        """
        Computes gradients of the target logits with respect to the input embeddings.

        Returns:
            A tensor of shape (batch_size, target_length, input_length) with gradients
            averaged over the embedding dimension.
        """
        model_inputs = self.embed(model_inputs)
        inputs_embeds = model_inputs["inputs_embeds"]

        def get_score(inputs_embeds: torch.Tensor):
            return self.get_targeted_logits(
                {"inputs_embeds": inputs_embeds, "attention_mask": model_inputs["attention_mask"]}, targets
            )

        # Compute gradient of the selected logits:
        grad_matrix = jacobian(get_score, inputs_embeds)  # (n, lt, n, l, d)
        grad_matrix = grad_matrix.mean(dim=-1)  # (n, lt, n, l)  # average over the embedding dimension
        n = grad_matrix.shape[0]  # number of examples in the batch
        diag_grad_matrix = grad_matrix[
            torch.arange(n), :, torch.arange(n), :
        ]  # (n, lt, l) # remove the gradients of the other examples

        return diag_grad_matrix

    @get_gradients.register(Iterable)
    def _(
        self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], targets: Iterable[torch.Tensor]
    ) -> Iterable[torch.Tensor]:
        """
        Computes gradients for a batch of model input-target pairs.
        """
        return [
            self.get_gradients(model_input, target) for model_input, target in zip(model_inputs, targets, strict=True)
        ]

    # @get_gradients.register(Iterable)
    # def _(
    #     self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], targets: Iterable[torch.Tensor]
    # ) -> Iterable[torch.Tensor]:
    #     model_inputs = [self.embed(model_input) for model_input in model_inputs]

    #     inputs_embeds = [model_input["inputs_embeds"] for model_input in model_inputs]
    #     attention_masks = [model_input["attention_mask"] for model_input in model_inputs]

    #     def get_score(inputs_embeds: torch.Tensor):
    #         model_inputs_aux = [
    #             {"inputs_embeds": inputs_embeds[i], "attention_mask": attention_masks[i]}
    #             for i in range(len(inputs_embeds))
    #         ]
    #         return self.get_targeted_logits(model_inputs_aux, targets)

    #     # Compute gradient of the selected logits:
    #     grad_matrix = jacobian(get_score, inputs_embeds)  # (n, lt, n, l, d)
    #     grad_matrix = grad_matrix.mean(dim=-1)  # (n, lt, n, l)  # average over the embedding dimension
    #     n = grad_matrix.shape[0]  # number of examples in the batch
    #     diag_grad_matrix = grad_matrix[
    #         torch.arange(n), :, torch.arange(n), :
    #     ]  # (n, lt, l) # remove the gradients of the other examples

    #     return diag_grad_matrix
