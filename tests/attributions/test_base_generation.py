from collections.abc import Mapping

import torch

from interpreto.attributions.base import GenerationAttributionExplainer


def create_targets_test(tokenizer):
    """
    Create a set of targets for testing the process_targets method.
    """
    # str:
    target1a = "I like kitten"
    target1b = "Interpreto is magic"

    # TensorMapping:
    target2a = tokenizer(target1a, return_tensors="pt")
    target2b = tokenizer([target1b], return_tensors="pt")

    # TensorMapping with multiple elements:
    target2c = tokenizer([target1a, target1b], return_tensors="pt", padding=True)

    # torch.Tensor:
    target3a = target2a["input_ids"]
    target3b = target2b["input_ids"]

    # list of str:
    target41 = [target1a, target1b]

    # list of TensorMapping:
    target42 = [target2a, target2b]
    target42c = [target2a, target2b, target2c]

    # list of torch.Tensor:
    target43 = [target3a, target3b]

    return [
        target1a,
        target2a,
        target3a,  # 3 first targets have 1 element
        target2c,
        target41,
        target42,
        target43,  # 4 next targets hace 2 elements
        target42c,  # the last targets has 4 elements
    ]


def test_process_targets(model, tokenizer):
    """
    Test the process_targets method for different input types.
    """
    explainer = GenerationAttributionExplainer(model, tokenizer, batch_size=2)
    list_targets = create_targets_test(tokenizer)

    for target in list_targets:
        results = explainer.process_targets(target)
        assert isinstance(results, list), "The output of the process_targets must be a list"
        assert all(isinstance(result, torch.Tensor) for result in results), (
            "The elements of the list must be of type torch.Tensor."
        )
        assert all(result.dim() == 2 for result in results), "The elements of the list must have 2 dimensions."
        assert all(result.shape[0] == 1 for result in results), (
            "The first dimension of the elements of the list must be 1."
        )


def test_process_inputs_to_explain_and_targets(model, tokenizer):
    explainer = GenerationAttributionExplainer(model, tokenizer, batch_size=2)  # type: ignore
    list_targets = create_targets_test(tokenizer)
    list_targets.append(None)  # Add None to the list of targets for testing

    # Model input example without special_tokens_mask
    model_input1 = tokenizer(["I like kittens and I like dogs."], return_tensors="pt")
    model_input2 = tokenizer(["Interpreto is incredible."], return_tensors="pt")

    model_input1element = [model_input1]
    model_input2elements = [model_input1, model_input2]
    model_input4elements = [model_input1, model_input2, model_input2, model_input1]
    model_inputs = (
        [model_input1element] * 3 + [model_input2elements] * 4 + [model_input4elements] + [model_input1element]
    )  # Repeat the model inputs for each target and the corresponding number of elements in the target (+ one for the None target).
    # TODO: add a raise error in the process_inputs_to_explain_and_targets method if the number of elements in the model_input is not equal to the number of elements in the target.

    for model_input, target in zip(model_inputs, list_targets, strict=True):
        processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
            model_input, targets=target
        )
        assert isinstance(processed_inputs, list), "processed_inputs must be a list"
        assert all(isinstance(processed_input, Mapping) for processed_input in processed_inputs), (
            "The elements of the processed_inputs list must be of type Mapping."
        )
        assert all(processed_input["input_ids"].dim() == 2 for processed_input in processed_inputs), (
            "The input_ids of the element of the processed_inputs list must have 2 dimensions."
        )
        assert all(processed_input["input_ids"].shape[0] == 1 for processed_input in processed_inputs), (
            "The first dimension of the input_ids of the element of the processed_inputs list must be 1."
        )
        assert all("special_tokens_mask" in processed_input for processed_input in processed_inputs), (
            "The processed_inputs must contain a 'special_tokens_mask' key."
        )

        assert isinstance(processed_targets, list), "processed_targets must be a list"
        assert all(isinstance(processed_target, torch.Tensor) for processed_target in processed_targets), (
            "The elements of the processed_targets list must be of type torch.Tensor."
        )
        assert all(processed_target.dim() == 2 for processed_target in processed_targets), (
            "The elements of the processed_targets list must have 2 dimensions."
        )
        assert all(processed_target.shape[0] == 1 for processed_target in processed_targets), (
            "The first dimension of the elements of the processed_targets list must be 1."
        )


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_process_targets(model, tokenizer)
test_process_inputs_to_explain_and_targets(model, tokenizer)
