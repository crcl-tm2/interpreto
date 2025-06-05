# Saliency

📰 [Paper](https://arxiv.org/abs/1312.6034)

Saliency maps are a simple and widely used gradient-based method for interpreting neural network predictions. The idea is to compute the gradient of the model's output with respect to its input embeddings to estimate which input tokens most influence the output.

Procedure:

- Pass the input through the model to obtain an output (e.g., class logit, token probability).
- Compute the gradient of the output with respect to the input embeddings.
- For each token, reduce the gradient vector (e.g., via norm with the embedding) to obtain a scalar importance score.

## Example

```python
from interpreto import Saliency

# load model, tokenizer and text
method = Saliency(model, tokenizer, batch_size=4)
explanations = method.explain(text)
```

## Parameters

TODO: use mkdocs built in functionalities to generate the table and put example in the docstring

- model (PreTrainedModel): Hugging Face model to explain
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
- batch_size (int): batch size for the attribution method
- inference_mode (InferenceModes): output space to analyse (e.g., InferenceModes.LOGITS, InferenceModes.SOFTMAX, InferenceModes.LOG_SOFTMAX)
- device (torch.device): device on which the attribution method will be run
