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
Tests for `interpreto.concepts.interpretation.llm_labels` methods
for `ConceptEncoderExplainer` and `ConceptAutoEncoderExplainer`
using the `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

import pytest
import torch

from interpreto.concepts.interpretations.llm_labels import (
    Example,
    _build_example_prompt,
    _format_examples,
    _sample_quantile,
    _sample_random,
    _sample_top,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def concept_activations() -> torch.Tensor:
    return torch.tensor([0.1, 0.5, 0.0, 8.5, 7.2, 0.0, 0.0, 1.4, 0.1, 3.8])


def test_sample_top(concept_activations: torch.Tensor):
    # Test output type and length
    selected_idx = _sample_top(
        concept_activations=concept_activations,
        k_examples=2,
    )
    assert isinstance(selected_idx, list)
    assert all(isinstance(id, int) for id in selected_idx)
    assert len(selected_idx) == 2
    assert 3 in selected_idx
    assert 4 in selected_idx

    # Test that 0 values are not kept
    selected_idx = _sample_top(
        concept_activations=concept_activations,
        k_examples=9,
    )
    assert isinstance(selected_idx, list)
    assert 2 not in selected_idx
    assert 5 not in selected_idx
    assert 6 not in selected_idx

    # Test that there is no repetition
    selected_idx = _sample_top(
        concept_activations=concept_activations,
        k_examples=15,
    )
    assert isinstance(selected_idx, list)
    assert len(set(selected_idx)) == len(selected_idx)

    # Test raising error
    with pytest.raises(ValueError):
        _sample_top(
            concept_activations=torch.rand(10, 5),
            k_examples=4,
        )


def test_sample_quantile(concept_activations: torch.Tensor):
    # Test output type and length
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=2,
        k_quantile=2,
    )
    assert isinstance(selected_idx, list)
    assert all(isinstance(id, int) for id in selected_idx)
    assert len(selected_idx) == 2

    # Test that 0 values are not kept
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=9,
        k_quantile=2,
    )
    assert isinstance(selected_idx, list)
    assert len(selected_idx) == 7  # min(k_examples//k_quantile * k_examples, num non zero samples)
    assert 2 not in selected_idx
    assert 5 not in selected_idx
    assert 6 not in selected_idx

    # Test that there is no repetition
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=15,
        k_quantile=2,
    )
    assert isinstance(selected_idx, list)
    assert len(set(selected_idx)) == len(selected_idx)

    # Test raising error
    with pytest.raises(ValueError):
        _sample_quantile(
            concept_activations=torch.rand(10, 5),
            k_examples=4,
            k_quantile=2,
        )
    with pytest.raises(ValueError):
        _sample_quantile(
            concept_activations=torch.rand(10, 5),
            k_examples=2,
            k_quantile=4,
        )

    # Test quantile
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=3,
        k_quantile=3,
    )
    assert len(selected_idx) == 3
    assert 3 in selected_idx or 4 in selected_idx  # 1st quantile
    assert 9 in selected_idx or 7 in selected_idx  # 2nd quantile
    assert 1 in selected_idx or 0 in selected_idx or 8 in selected_idx  # 3rd quantile


def test_sample_random(concept_activations: torch.Tensor):
    # Test output type and length
    selected_idx = _sample_random(
        concept_activations=concept_activations,
        k_examples=2,
    )
    assert isinstance(selected_idx, list)
    assert all(isinstance(id, int) for id in selected_idx)
    assert len(selected_idx) == 2

    # Test that 0 values are not kept
    selected_idx = _sample_random(
        concept_activations=concept_activations,
        k_examples=9,
    )
    assert isinstance(selected_idx, list)
    assert 2 not in selected_idx
    assert 5 not in selected_idx
    assert 6 not in selected_idx

    # Test that there is no repetition
    selected_idx = _sample_random(
        concept_activations=concept_activations,
        k_examples=15,
    )
    assert isinstance(selected_idx, list)
    assert len(set(selected_idx)) == len(selected_idx)

    # Test raising error
    with pytest.raises(ValueError):
        _sample_random(
            concept_activations=torch.rand(10, 5),
            k_examples=4,
        )


def test_format_examples():
    examples = _format_examples(
        example_ids=[3, 2, 6],
        inputs=["This", " is", " a", " test", "Another", " sentence", " with", " more", " words"],
        concept_activations=torch.tensor([0.1, 0.0, 8.5, 5.3, 7.2, 0.0, 4.2, 0.0, 0.1]),
        sample_ids=[0, 0, 0, 0, 1, 1, 1, 1, 1],
        k_context=1,  # 1 on the right and 1 on the left
    )
    assert isinstance(examples, list)
    assert len(examples) == 3
    assert examples[0].texts == [" a", " test"]
    assert examples[0].activations[0] == 10
    assert examples[0].activations[1] == 6
    assert examples[1].texts == [" is", " a", " test"]
    assert examples[2].texts == [" sentence", " with", " more"]
    # Test sentences without context
    examples = _format_examples(
        example_ids=[0, 2],
        inputs=[
            "This is a test with sentences",
            "Another sentence with more words",
            "And another one",
        ],
        concept_activations=torch.tensor([8.5, 0.0, 2.1]),
        sample_ids=[0, 1, 2],
        k_context=0,  # no context
    )
    assert isinstance(examples, list)
    assert len(examples) == 2
    assert examples[0].texts == "This is a test with sentences"
    assert isinstance(examples[0].activations, int)
    assert examples[0].activations == 10
    assert examples[1].texts == "And another one"
    assert isinstance(examples[1].activations, int)


def test_build_example_prompt():
    # Test with examples without context
    examples = [
        Example(
            texts="This is a test with sentences",
            activations=10,
        ),
        Example(
            texts="This is another sentence",
            activations=4,
        ),
    ]
    prompt = _build_example_prompt(examples)
    assert isinstance(prompt, str)
    assert (
        prompt
        == "Example 1: This is a test with sentences (activation: 10)\nExample 2: This is another sentence (activation: 4)"
    )

    # Test with examples with context
    examples = [
        Example(
            texts=["This", " is", " a", " test"],
            activations=[2, 6, 10, 0],
        ),
        Example(
            texts=["Another", " sentence", " with", " more", " words"],
            activations=[0, 2, 4, 0, 0],
        ),
    ]
    prompt = _build_example_prompt(examples)
    assert isinstance(prompt, str)
    assert (
        prompt
        == 'Example 1: This is << a>>  test\nActivations: ("This", 2), (" is", 6), (" a", 10), (" test", 0)\nExample 2: Another sentence << with>>  more words\nActivations: ("Another", 0), (" sentence", 2), (" with", 4), (" more", 0), (" words", 0)'
    )


# def test_llm_labels_concept_selection(splitted_encoder_ml: ModelWithSplitPoints):
#     """
#     Test that the `interpret` method works as expected
#     Fake activations are given to the `NeuronsAsConcepts` explainer
#     """

#     # initializing the explainer
#     hidden_size = 32
#     split = "bert.encoder.layer.1.output"
#     splitted_encoder_ml.split_points = split
#     concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split).concept_model

#     # initializing the interpreter
#     interpretation_method = LLMLabels(
#         model_with_split_points=splitted_encoder_ml,
#         split_point=split,
#         concept_model=concept_model,
#         activation_granularity=ActivationGranularity.TOKEN,
#         llm_interface=None, #TODO : mock this
#         sampling_method=SAMPLING_METHOD.TOP,
#         k_examples=2,
#     )

#     labels = interpretation_method.interpret(
#         concepts_indices=[0, 5, 13],
#         inputs=["a sentence", "another sentence", "yet another sentence"],
#     )
#     assert isinstance(labels, dict)
#     assert len(labels) == 3
#     assert 5 in labels
#     assert 0 in labels
#     assert 13 in labels

#     labels = interpretation_method.interpret(
#         concepts_indices=0
#         inputs=["a sentence", "another sentence", "yet another sentence"],
#     )
#     assert 0 in labels
#     assert len(labels) == 1


# @pytest.mark.parametrize(
#     "activation_granularity",
#     [
#         ModelWithSplitPoints.activation_granularities.TOKEN,
#         ModelWithSplitPoints.activation_granularities.WORD,
#         ModelWithSplitPoints.activation_granularities.SENTENCE,
#         ModelWithSplitPoints.activation_granularities.SAMPLE,
#     ],
# )
# def test_llm_labels_granularity(
#     splitted_encoder_ml: ModelWithSplitPoints, huge_text: list[str], activation_granularity: ActivationGranularity
# ):
#     """
#     Test that the `_sample_examples` method works as expected for the different granularities
#     Fake activations are given to the `NeuronsAsConcepts` explainer
#     """
#     # initializing the explainer
#     split = "bert.encoder.layer.1.output"
#     splitted_encoder_ml.split_points = split
#     concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split).concept_model

#     # getting the activations
#     activations = splitted_encoder_ml.get_activations(huge_text, activation_granularity=activation_granularity)[split]

#     # initializing the interpreter
#     interpretation_method = LLMLabels(
#         model_with_split_points=splitted_encoder_ml,
#         split_point=split,
#         concept_model=concept_model,
#         activation_granularity=ActivationGranularity.TOKEN,
#         llm_interface=None, #TODO : mock this
#         sampling_method=SAMPLING_METHOD.TOP,
#         k_examples=2,
#     )

#     # extracting concept interpretations
#     examples = interpretation_method._sample_examples(
#         inputs=words_list_list,
#         concepts_activations=fake_activations,
#         sample_ids=list(range(nb_concepts)),
#         concept_idx=0,
#     )

#     flattened_huge_text = ". ".join(huge_text).replace("\n", " ")

#     assert isinstance(topk_inputs, dict) and len(topk_inputs) == 3
#     for c in [0, 5, 13]:
#         assert c in topk_inputs
#         assert len(topk_inputs[c]) == 2
#         for key in topk_inputs[c].keys():
#             new_key = key[2:] if key.startswith("##") else key
#             assert new_key.lower().replace("\n", " ") in flattened_huge_text.lower()


# def test_llm_labels_sources(splitted_encoder_ml: ModelWithSplitPoints):
#     """
#     Test that different sources give the same results for top and quantile sampling.
#     """
#     hidden_size = 32
#     n_tokens = 6
#     k = 3
#     n_samples = k * 5
#     assert k <= n_tokens  # otherwise the test will break

#     # generating data with duplicates and different lengths
#     joined_tokens_list = [" ".join([f"{i + j}" for j in range(n_tokens)]) for i in range(n_samples)]
#     larger_input = " ".join(["test" for _ in range(2 * n_tokens)])
#     joined_tokens_list.append(larger_input)

#     # initializing the explainer
#     split = "bert.encoder.layer.1.output"
#     splitted_encoder_ml.split_points = split
#     concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split)

#     # getting the activations
#     activations = splitted_encoder_ml.get_activations(
#         joined_tokens_list, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
#     )

#     llm_labels_interpreter = LLMLabels(
#         model_with_split_points=splitted_encoder_ml,
#         split_point=split,
#         concept_model=concept_model,
#         activation_granularity=ActivationGranularity.TOKEN,
#         llm_interface=None, #TODO : mock this
#         sampling_method=SAMPLING_METHOD.TOP,
#         k_examples=2,
#     )

#     # getting the top k tokens
#     samples_from__inputs = llm_labels_interpreter._sample_examples(
#         inputs=joined_tokens_list,
#         concepts_activations=activations,
#         sample_ids=list(range(hidden_size)),
#         concept_idx=0,
#     )
#     samples_from_latent = concept_explainer.interpret(
#         interpretation_method=LLMLabels,
#         activation_granularity=ActivationGranularity.TOKEN,
#         concepts_indices="all",
#         inputs=joined_tokens_list,
#         latent_activations=activations,
#     )
#     samples_from_concept = concept_explainer.interpret(
#         interpretation_method=LLMLabels,
#         activation_granularity=ActivationGranularity.TOKEN,
#         concepts_indices="all",
#         inputs=joined_tokens_list,
#         concepts_activations=activations[split],
#     )

#     assert list(range(hidden_size)) == list(top_k_latent.keys())
#     assert list(range(hidden_size)) == list(top_k_concept.keys())
#     assert list(range(hidden_size)) == list(top_k_inputs.keys())

#     for top_latent, top_concept, top_input in zip(
#         top_k_latent.values(), top_k_concept.values(), top_k_inputs.values(), strict=True
#     ):
#         assert len(top_latent) == k
#         assert top_latent == top_concept == top_input


# def test_llm_labels_from_vocabulary(splitted_encoder_ml: ModelWithSplitPoints):
#     """
#     Test that interpretations can be obtained from the vocabulary
#     """
#     k = 2
#     hidden_size = 32
#     nb_concepts = 3

#     # initializing the explainer
#     split = "bert.encoder.layer.1.output"
#     splitted_encoder_ml.split_points = split
#     concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split)

#     top_k_vocabulary = concept_explainer.interpret(
#         interpretation_method=TopKInputs,
#         activation_granularity=TopKInputs.activation_granularities.TOKEN,
#         k=k,
#         concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
#         use_vocab=True,
#     )

#     assert len(top_k_vocabulary) == nb_concepts

#     vocabulary = list(splitted_encoder_ml.tokenizer.get_vocab().keys())
#     for topk in top_k_vocabulary.values():
#         assert len(topk) == k
#         for token in topk:
#             assert token in vocabulary


# def test_llm_labels_error_raising(
#     splitted_encoder_ml: ModelWithSplitPoints, activations_dict: dict[str, torch.Tensor]
# ):
#     """
#     Test that the `TopKInputs` class raises an error when needed
#     """
#     # getting the activations
#     activations = next(iter(activations_dict.values()))  # dictionary with only one element

#     concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml).concept_model

#     some_texts_for_interpretation = ["a sentence", "another sentence", "yet another sentence"]

#     # When use_vocab=False and inputs is not provided
#     with pytest.raises(ValueError):
#         method = TopKInputs(
#             model_with_split_points=splitted_encoder_ml,
#             concept_model=concept_model,
#             activation_granularity=TopKInputs.activation_granularities.TOKEN,
#             use_vocab=False,
#         )
#         method.interpret(
#             concepts_indices=0,
#         )

#     # wrong indices
#     for wrong_indices in [-1, activations.shape[1], [-1, 0, 1], ["?"], (0, 1, 2), "str other than 'all'"]:
#         with pytest.raises(ValueError):
#             method = TopKInputs(
#                 model_with_split_points=splitted_encoder_ml,
#                 concept_model=concept_model,
#                 activation_granularity=TopKInputs.activation_granularities.TOKEN,
#             )
#             method.interpret(
#                 concepts_indices=wrong_indices,
#                 inputs=some_texts_for_interpretation,
#                 concepts_activations=activations,
#             )


# if __name__ == "__main__":
#     from transformers import AutoModelForMaskedLM

#     test_sample_top()
#     test_sample_quantile()
#     test_add_context()
#     test_build_prompt()

#     splitted_encoder_ml = ModelWithSplitPoints(
#         "hf-internal-testing/tiny-random-bert",
#         split_points=["bert.encoder.layer.1.output"],
#         model_autoclass=AutoModelForMaskedLM,  # type: ignore
#     )
#     sentences = [
#         "Lorem ipsum dolor sit amet, consectetur adipiscing elit. sed do eiusmod tempor incididunt\n\nut labore et dolore magna aliqua.",
#         "Interpreto is magical",
#         "Testing interpreto",
#     ]
#     activation_dict = splitted_encoder_ml.get_activations(
#         sentences, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
#     )

#     test_llm_labels_concept_selection(splitted_encoder_ml)
#     test_llm_labels_granularity(splitted_encoder_ml, sentences * 10, ModelWithSplitPoints.activation_granularities.SAMPLE)
#     test_llm_labels_sources(splitted_encoder_ml)
#     test_llm_labels_from_vocabulary(splitted_encoder_ml)
#     test_llm_labels_error_raising(splitted_encoder_ml, activation_dict)
