from itertools import product

import torch

from interpreto.attributions import (
    IntegratedGradients,
)

from ...fixtures.model_zoo import (
    SimpleTokenizer,
    SmallConv1DModel,
    SmallConv2DModel,
    SmallDenseModel,
    SmallGRUModel,
    SmallLSTMModel,
    SmallRNNModel,
    SmallTextClassifier,
    SmallTextGenerator,
    SmallViTModel,
)

attribution_methods_to_test = [IntegratedGradients]


def test_attribution_methods_with_1d_input_models():
    print("\nTesting 1D Input Models...")

    input_shape = (4, 10)  # (batch_size=4, input_size=10)
    model = SmallDenseModel(input_size=input_shape[1])
    input_tensor = torch.randn(input_shape)

    for attribution_explainer in attribution_methods_to_test:
        attributions = attribution_explainer(model).explain(input_tensor)

        assert attributions.shape == input_shape


def test_attribution_methods_with_2d_input_models():
    print("\nTesting 2D Input Models...")

    input_shape = (4, 3, 10)  # (batch_size=4, channels=3, input_size=10)
    models = [
        SmallRNNModel(input_size=input_shape[2]),
        SmallLSTMModel(input_size=input_shape[2]),
        SmallGRUModel(input_size=input_shape[2]),
        SmallConv1DModel(input_channels=input_shape[1]),
    ]

    input_tensor = torch.randn(input_shape)

    for model, attribution_explainer in product(models, attribution_methods_to_test):
        attributions = attribution_explainer(model).explain(input_tensor)

        assert attributions.shape == input_shape


def test_attribution_methods_with_3d_input_models():
    print("\nTesting 3D Input Models...")

    input_shape = (4, 3, 10, 5)  # (batch_size=4, channels=3, input_h=10, input_w=5)
    models = [
        SmallConv2DModel(input_channels=input_shape[1]),
        SmallViTModel(image_size=input_shape[2], input_size=(input_shape[2], input_shape[3])),
    ]

    input_tensor = torch.randn(input_shape)

    for model, attribution_explainer in product(models, attribution_methods_to_test):
        attributions = attribution_explainer(model).explain(input_tensor)

        assert attributions.shape == input_shape


def test_attribution_methods_with_text_classifier():
    print("\nTesting Text Classifier...")

    tokenizer = SimpleTokenizer()
    classifier = SmallTextClassifier()

    input_text = "word1 word5 word20"
    tokenized_text = tokenizer.encode(input_text).unsqueeze(0)  # (batch_size=1, seq_len)

    for attribution_explainer in attribution_methods_to_test:
        attributions = attribution_explainer(classifier).explain(tokenized_text)

        assert attributions.shape == torch.Size([1, 10])


def test_attribution_methods_with_text_generator():
    print("\nTesting Text Generator...")

    tokenizer = SimpleTokenizer()
    generator = SmallTextGenerator()

    input_text = "word1 word5 word20"
    tokenized_text = tokenizer.encode(input_text).unsqueeze(0)  # (batch_size=1, seq_len)

    for attribution_explainer in attribution_methods_to_test:
        attributions = attribution_explainer(generator).explain(tokenized_text, tokenized_text)

        assert attributions.shape == torch.Size([1, 20, 10])
