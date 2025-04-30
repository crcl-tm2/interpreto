# Occlusion

📰 [Paper](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)

The Occlusion method is a perturbation-based approach to interpret model behavior by analyzing the impact of removing or masking parts of the input text. 
The principle is simple: by systematically occluding (i.e., masking, deleting, or replacing) specific tokens or spans in the input and observing how the model's output changes, one can infer the relative importance of each part of the input to the model's behavior.

## Example

```python
from interpreto.attributions import OcclusionExplainer
from interpreto.commons.granularity import GranularityLevel

# load model, tokenizer and text
method = OcclusionExplainer(model, tokenizer, batch_size=4, granularity_level=GranularityLevel.WORD)
explanations = method.explain(text)
```

## Parameters

- model (PreTrainedModel): Hugging Face model to explain
- tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
- batch_size (int): batch size for the attribution method
- granularity_level (GranularityLevel): granularity level of the perturbations (token, word, sentence, etc.)
- device (torch.device): device on which the attribution method will be run
- replace_token_id (int): token id to use for replacing the occluded tokens
