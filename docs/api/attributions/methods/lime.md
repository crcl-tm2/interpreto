# LIME

📰 [Paper](https://dl.acm.org/doi/abs/10.1145/2939672.2939778)

Local Interpretable Model‑agnostic Explanations (LIME) is a perturbation‑based approach that explains individual predictions by fitting a simple, interpretable surrogate model locally around the prediction of interest. By sampling perturbed versions of the input and weighting them by their proximity to the original instance, LIME learns per‑feature importance scores that approximate the behaviour of the underlying black‑box model in that local region.

## Example

```python
from interpreto import Granularity, Lime
from interpreto.attributions import InferenceModes

# load model, tokenizer and text
method = Lime(model, tokenizer, batch_size=4, 
    inference_mode=InferenceModes.LOG_SOFTMAX,
    n_perturbations=20,
    granularity=Granularity.WORD,
    distance_function=Lime.distance_functions.HAMMING)
explanations = method(text)
```

## Parameters

TODO: use mkdocs built in functionalities to generate the table and put example in the docstring

- model (PreTrainedModel): Hugging Face model to explain  
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model  
- batch_size (int): batch size for the attribution method  
- inference_mode (InferenceModes): output space to analyse (e.g., InferenceModes.LOGITS, InferenceModes.SOFTMAX, InferenceModes.LOG_SOFTMAX)  
- n_perturbations (int): number of perturbed samples used to train the local surrogate model  
- granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)  
- distance_function (Callable): function to compute similarity between original and perturbed samples (e.g., Lime.distance_functions.HAMMING)  
- device (torch.device): device on which the attribution method will be run  

