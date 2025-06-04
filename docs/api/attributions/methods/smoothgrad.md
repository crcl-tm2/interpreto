# Smoothgrad

📰 [Paper](https://arxiv.org/abs/1706.03825)

SmoothGrad is an enhanced version of gradient-based interpretability methods, such as saliency maps. It reduces the noise and visual instability often seen in raw gradient attributions by averaging gradients over multiple noisy versions of the input. The result is a smoothed importance score for each token.

Procedure:

- Generate multiple perturbed versions of the input by adding noise (Gaussian) to the input embeddings.
- For each noisy input, compute the gradient of the output with respect to the embeddings.
- Average the gradients across all samples.
- Aggregate the result per token (e.g., by norm with the input) to get the final attribution scores.

## Example

```python
from interpreto.attributions import Smoothgrad

# load model, tokenizer and text
method = Smoothgrad(model, tokenizer, batch_size=4, n_interpolations=50, noise_level=0.01)
explanations = method.explain(text)
```

## Parameters

TODO: use mkdocs built in functionalities to generate the table and put example in the docstring

- model (PreTrainedModel): Hugging Face model to explain.
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model.
- batch_size (int): batch size for the attribution method.
- device (torch.device): device on which the attribution method will be run.
- n_interpolations (int): the number of interpolations to generate multiple perturbed versions of the input by adding noise.
- noise_level (float): standard deviation of the Gaussian noise to add to the inputs.
