# Integrated Gradients

📰 [Paper](https://arxiv.org/abs/1703.01365)

Integrated Gradients (IG) is a gradient-based interpretability method that attributes importance scores to input features (e.g., tokens) by integrating the model’s gradients along a path from a baseline input to the actual input. The method is designed to address some of the limitations of standard gradients, such as saturation and noise, by averaging gradients over interpolated inputs rather than relying on a single local gradient.

Procedure:

- Choose a baseline input (e.g., a sequence of padding tokens, [MASK] tokens, or zeros) representing the absence of information.
- Linearly interpolate inputs between the baseline and the actual input over n steps.
- Compute the gradient of the model’s output with respect to the input embeddings at each step.
- Approximate the integral of the gradients along the path and multiply by the input difference to get the attribution scores.

## Example

```python
from interpreto.attributions import IntegratedGradients

# load model, tokenizer and text
method = IntegratedGradients(model=model, tokenizer=tokenizer, batch_size=4, n_interpolations=50)
explanations = method.explain(model_inputs = text)
```

## Parameters

TODO: use mkdocs built in functionalities to generate the table and put example in the docstring

- model (PreTrainedModel): Hugging Face model to explain.
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model.
- batch_size (int): batch size for the attribution method.
- device (torch.device): device on which the attribution method will be run.
- n_interpolations (int): Number of points to interpolate between the baseline and the desired point.
- baseline (torch.Tensor | float | None): the baseline to use for the interpolations.
