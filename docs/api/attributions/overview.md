# API: Attributions Methods

## Common API

```
from interpreto import Method

explainer = Method(model, tokenizer, device, args)
explanation = explainer(inputs, targets)
```

The API have two steps:

- explainer instantiation: Method is an attribution method among those displayed methods tables. It inherits from the Base class BlackBoxExplainer. Their initialization takes 2 parameters apart from the specific ones and generates an explainer:
  - model: the model from which we want to obtain attributions.
  - tokenizer: if the model is an NLP model, the associated tokenizer must be given, otherwise None.
  - batch_size: an integer which allows to process inputs per batch.
  - args: specifics for each method.

- explain method: The call to explainer generates the explanations, it takes two parameters:
  - inputs: the samples on which the explanations are requested, see inputs section for more detail.
  - targets: another parameter to specify what to explain in the inputs, can be a specific class or a set of classes (for classification), or texts (for generation). If targets=None, the target is calculated by making the prediction from the input with the given model.

## Specifics methods

**Inference-based Methods:**

- [Kernel SHAP](./docs/attributions/methods/kernelshap.md): [Lundberg and Lee, 2017, A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
- [LIME](./docs/attributions/methods/lime.md): [Ribeiro et al. 2013, "Why should i trust you?" explaining the predictions of any classifier](https://dl.acm.org/doi/abs/10.1145/2939672.2939778).
- [Occlusion](./docs/attributions/methods/occlusion.md): [Zeiler and Fergus, 2014. Visualizing and understanding convolutional networks](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53).
- [Sobol Attribution](./docs/attributions/methods/sobol.md): [Fel et al. 2021, Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html).

**Gradient based methods:**

- [Integrated Gradient](./docs/attributions/methods/integrated_gradients.md): [Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks](http://proceedings.mlr.press/v70/sundararajan17a.html).
- [Saliency](./docs/attributions/methods/saliency.md): [Simonyan et al. 2013, Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034).
- [SmoothGrad](./docs/attributions/methods/smoothgrad.md): [Smilkov et al. 2017, SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)

# Custom API

Although we've created an API that allows the user to call specific allocation methods directly, a harmonized API for all allocation methods can be used for greater freedom.

```
perturbator = Perturbator(inputs_embedder)
aggregator = Aggregator()
explainer = AttributionExplainer(model, tokenizer, batch_size, perturbator, aggregator, device)
explanation = explainer(inputs, targets)
```

The API have two steps:

First step:
Instantiate a perturbator and an aggregator that will be used by the attribution method.

perturbator:
    - inputs_embedder: Optional module used to embed input IDs when only ``input_ids`` are provided.

aggregator: no argument.


Second step:
Instantiate the AttributionExplainer class with the perturbator and aggregator.

- AttributionExplainer is the class that instantiates any attribution method based on whether or not it uses gradients (with the boolean argument use_gradient), its perturbation type and its aggregation type. It take the custom parameters (perturbator, aggregator, use_gradient) and the classic parameters for explaination:
  - model (PreTrainedModel): Hugging Face model to explain,
  - tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model,
  - batch_size (int): batch size for the attribution method,
  - perturbator (BasePerturbator | None = None): Instance used to generate input perturbations.
                If None, the perturbator returns only the original input,
  - aggregator (Aggregator | None = None): Instance used to aggregate computed attribution scores.
                If None, the aggregator returns the original scores,
  - device (torch.device | None = None): device on which the attribution method will be run,
  - use_gradient: boolean.

- La classe AttributionExplainer, contient une méthode explain: The call to explainer generates the explanations, it takes two parameters:
  - inputs: the samples on which the explanations are requested, see inputs section for more detail.
  - targets: another parameter to specify what to explain in the inputs, can be a specific class or a set of classes (for classification), or texts (for generation).
