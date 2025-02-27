from itertools import product

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

from interpreto.attributions import (
    IntegratedGradients,
)

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


# def test_attribution_methods_with_text_classifier():
#     tokenizer = SimpleTokenizer()
#     classifier = SmallTextClassifier()

#     input_text = "word1 word5 word20"
#     tokenized_text = tokenizer.encode(input_text).unsqueeze(0)  # (batch_size=1, seq_len)

#     for attribution_explainer in attribution_methods_to_test:
#         attributions = attribution_explainer(classifier, batch_size=3).explain(tokenized_text)

#         assert attributions.shape == torch.Size([1, 10])


# def test_attribution_methods_with_text_generator():  # TODO: add this test
#     tokenizer = SimpleTokenizer()
#     generator = SmallTextGenerator()

#     input_text = "word1 word5 word20"
#     tokenized_text = tokenizer.encode(input_text).unsqueeze(0)  # (batch_size=1, seq_len)

#     for attribution_explainer in attribution_methods_to_test:
#         attributions = attribution_explainer(generator).explain(
#             tokenized_text, tokenized_text
#         )  # TODO: check if this is correct

#         assert attributions.shape == torch.Size([1, 20, 10])
