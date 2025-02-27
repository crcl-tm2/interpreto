import torch
from test.fixtures.model_zoo import SmallDenseModel

from interpreto.attributions import IntegratedGradients


def test_tensor_integrated_gradients():
    model = SmallDenseModel()
    model.eval()
    ig = IntegratedGradients(model, n_samples=10)
    attributions = ig.attribute(input, target)
    assert attributions.shape == torch.Size([2, 1])


def test_token_embeddings_integrated_gradients():
    pass
