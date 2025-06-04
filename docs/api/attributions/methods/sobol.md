# Sobol Attribution

📰 [Paper](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html)


Sobol is a variance-based sensitivity analysis method used to quantify the contribution of each input component to the output variance of the model.  
It estimates both the first-order (main) and total (interaction) effects of features using Monte Carlo sampling strategies. In NLP, Sobol helps assess which words or tokens are most influential for the model’s decision, including how they interact with one another.

## Example

```python
from interpreto import Granularity, Sobol
from interpreto.attributions import InferenceModes

# load model, tokenizer and text
method = Sobol(model, tokenizer, batch_size=4, 
    inference_mode=InferenceModes.LOGITS,
    n_token_perturbations=8,
    granularity=Granularity.WORD)
explanations = method(text)
```

## Parameters

TODO: use mkdocs built in functionalities to generate the table and put example in the docstring

- model (PreTrainedModel): Hugging Face model to explain  
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model  
- batch_size (int): batch size for the attribution method  
- granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)  
- inference_mode (InferenceModes): output space to analyse (e.g., InferenceModes.LOGITS, InferenceModes.SOFTMAX, InferenceModes.LOG_SOFTMAX)  
- n_token_perturbations (int): number of perturbation samples used to estimate Sobol indices  
- sobol_indices_order (SobolIndicesOrders): order of the Sobol indices to estimate, either `FIRST_ORDER` or `TOTAL_ORDER`  
- sampler (SequenceSamplers): sampling method used to generate perturbations, one of `SOBOL`, `HALTON`, or `LATIN_HYPERCUBE`  
- device (torch.device): device on which the attribution method will be run  
