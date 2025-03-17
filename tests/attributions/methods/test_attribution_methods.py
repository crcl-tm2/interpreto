from itertools import product

import pytest
import torch
from nnsight import NNsight
from tests.fixtures.model_zoo import (
    # SimpleTokenizer,
    SmallConv1DModel,
    SmallConv2DModel,
    SmallDenseModel,
    SmallGRUModel,
    SmallLSTMModel,
    SmallRNNModel,
    # SmallTextClassifier,
    # SmallTextGenerator,
    SmallViTModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    # AutoModelForMultipleChoice,
    # AutoModelForQuestionAnswering,
    # AutoModelForTokenClassification,
    AutoTokenizer,
)

from interpreto.attributions import (
    IntegratedGradients,
)
from interpreto.attributions.base import AttributionOutput

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

attribution_methods_to_test = [IntegratedGradients]


def test_attribution_methods_with_1d_input_models():
    input_shape = (4, 10)  # (batch_size=4, input_size=10)
    model = NNsight(SmallDenseModel(input_size=input_shape[1]))
    input_tensor = torch.randn(input_shape)

    for attribution_explainer in attribution_methods_to_test:
        attributions = attribution_explainer(model, batch_size=3, device=DEVICE).explain(input_tensor)

        assert attributions.shape == input_shape


def test_attribution_methods_with_2d_input_models():
    input_shape = (4, 3, 10)  # (batch_size=4, channels=3, input_size=10)
    models = [
        NNsight(SmallRNNModel(input_size=input_shape[2])),
        NNsight(SmallLSTMModel(input_size=input_shape[2])),
        NNsight(SmallGRUModel(input_size=input_shape[2])),
        NNsight(SmallConv1DModel(input_channels=input_shape[1], input_length=input_shape[2])),
    ]

    input_tensor = torch.randn(input_shape)

    for model, attribution_explainer in product(models, attribution_methods_to_test):
        attributions = attribution_explainer(model, batch_size=3, device=DEVICE).explain(input_tensor)
        assert attributions.shape == input_shape


def test_attribution_methods_with_3d_input_models():
    input_shape = (4, 3, 10, 5)  # (batch_size=4, channels=3, input_h=10, input_w=5)
    models = [
        NNsight(SmallConv2DModel(input_channels=input_shape[1], image_size=(input_shape[2], input_shape[3]))),
        NNsight(SmallViTModel(input_channels=input_shape[1], image_size=(input_shape[2], input_shape[3]))),
    ]

    input_tensor = torch.randn(input_shape)

    for model, attribution_explainer in product(models, attribution_methods_to_test):
        attributions = attribution_explainer(model, batch_size=3, device=DEVICE).explain(input_tensor)

        assert attributions.shape == input_shape


model_loader_combinations = [
    ("hf-internal-testing/tiny-random-DebertaV2Model", AutoModelForSequenceClassification),
    ("hf-internal-testing/tiny-random-DebertaV2Model", AutoModelForMaskedLM),
    ("hf-internal-testing/tiny-random-xlm-roberta", AutoModelForSequenceClassification),
    ("hf-internal-testing/tiny-random-xlm-roberta", AutoModelForMaskedLM),
    ("hf-internal-testing/tiny-random-DistilBertModel", AutoModelForSequenceClassification),
    ("hf-internal-testing/tiny-random-DistilBertModel", AutoModelForMaskedLM),
    ("hf-internal-testing/tiny-random-t5", AutoModelForSequenceClassification),
    ("hf-internal-testing/tiny-random-t5", AutoModelForSeq2SeqLM),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", AutoModelForCausalLM),
    ("hf-internal-testing/tiny-random-gpt2", AutoModelForCausalLM),
]
# Currently supported: AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM.
# To do later:
# list_load_model = [
#     AutoModelForMultipleChoice,
#     AutoModelForQuestionAnswering,
#     AutoModelForTokenClassification,
# ]

all_combinations = list(product(model_loader_combinations, attribution_methods_to_test))


@pytest.mark.parametrize("model_name, model_loader, attribution_explainer", all_combinations)
def test_attribution_methods_with_text(model_name, model_loader, attribution_explainer):
    """Tests all combinations of models and loaders with an attribution method"""
    input_text = [
        "I love this movie!",
        "You are the best",
        "The cat is on the mat.",
        "My preferred film is Titanic",
        "Sorry, I am late,",
    ]

    try:
        model = model_loader.from_pretrained(model_name).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        assert model is not None, f"Model loading failed {model_name}"
        assert tokenizer is not None, f"Tokenizer failed to load for {model_name}"

        # To be changed according to the final form of the explainer:
        attributions = attribution_explainer(model, tokenizer=tokenizer, batch_size=3, device=DEVICE).explain(
            input_text
        )

        # Checks:
        assert isinstance(attributions, list), "The output of the attribution explainer must be a list"
        assert len(attributions) == len(input_text), (
            "The number of elements in the list must correspond to the number of inputs."
        )
        assert all(isinstance(attribution, AttributionOutput) for attribution in attributions), (
            "The elements of the list must be of type AttributionOutput."
        )
        assert all(len(attribution.elements) == len(attribution.attributions) for attribution in attributions), (
            "In the AttributionOutput class, elements and attributions must have the same length."
        )

    except Exception as e:
        pytest.fail(
            f"The test failed for {model_name} with {model_loader} and {attribution_explainer.__class__.__name__}: {str(e)}"
        )
