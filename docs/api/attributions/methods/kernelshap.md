# KernelSHAP

📰 [Paper](https://arxiv.org/abs/1705.07874)

KernelSHAP is a model‑agnostic Shapley value estimator that interprets predictions by computing Shapley values through a weighted linear regression in the space of feature coalitions. By unifying ideas from LIME and Shapley value theory, KernelSHAP provides additive feature attributions with strong consistency guarantees.


## Example

```python
from interpreto import Granularity, KernelShap
from interpreto.attributions import InferenceModes

# load model, tokenizer and text
method = KernelShap(model, tokenizer, batch_size=4, 
    inference_mode=InferenceModes.SOFTMAX,
    n_perturbations=20,
    granularity=Granularity.WORD)
explanations = method(text)
```

## Parameters

TODO: use mkdocs built in functionalities to generate the table and put example in the docstring

- model (PreTrainedModel): Hugging Face model to explain  
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model  
- batch_size (int): batch size for the attribution method  
- inference_mode (InferenceModes): output space to analyse (e.g., InferenceModes.LOGITS, InferenceModes.SOFTMAX, InferenceModes.LOG_SOFTMAX)  
- n_perturbations (int): number of perturbations (coalitions) used to estimate Shapley values  
- granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)  
- device (torch.device): device on which the attribution method will be run  

