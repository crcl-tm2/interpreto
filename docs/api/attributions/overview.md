# API: Attributions Methods

## Common API

```
explainer = Methods(model, tokenizer, device, args)
explanation = explainer.explain(inputs, targets)
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
- Occlusion: [Zeiler and Fergus, 2014. Visualizing and understanding convolutional networks](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53).
- LIME: [Ribeiro et al. 2013, "Why should i trust you?" explaining the predictions of any classifier](https://dl.acm.org/doi/abs/10.1145/2939672.2939778).
- Kernel SHAP: [Lundberg and Lee, 2017, A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
- Sobol Attribution: [Fel et al. 2021, Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html).

**Gradient based methods:**
- Saliency, InputxGradient: [Simonyan et al. 2013, Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034).
- Integrated Gradient: [Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks](http://proceedings.mlr.press/v70/sundararajan17a.html).
- DeepLift: [Shrikumar et al. 2017, Learning Important Features Through Propagating Activation Differences](http://proceedings.mlr.press/v70/shrikumar17a).
- SmoothGrad: [Smilkov et al. 2017, SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
- VarGrad: [Richter et al. 2020, VarGrad: A Low-Variance Gradient Estimator for Variational Inference](https://proceedings.neurips.cc/paper/2020/hash/9c22c0b51b3202246463e986c7e205df-Abstract.html)



# Custom API

Although we've created an API that allows the user to call specific allocation methods directly, a harmonized API for all allocation methods can be used for greater freedom.

```
inference_wrapper = InferenceWrapper(model, batch_size)
perturbator = Perturbator(tokenizer, inputs_embedder)
aggregator = Aggregator(todo)
explainer = AttributionExplainer(inference_wrapper, perturbator, aggregator, device)
explanation = explainer.explain(inputs, targets)
```


The API have two steps:

inference_wrapper:
    - model: the model from which we want to obtain attributions.
    - batch_size: an integer which allows to process inputs per batch.

perturbator:
    -

- explainer instantiation: Method is an attribution method among those displayed methods tables. It inherits from the Base class BlackBoxExplainer. Their initialization takes 2 parameters apart from the specific ones and generates an explainer:
    - inference_wrapper: Callable,
    - perturbator: BasePerturbator | None = None,
    - aggregator: Aggregator | None = None,
    - device: torch.device | None = None

- explain method: The call to explainer generates the explanations, it takes two parameters:
    - inputs: the samples on which the explanations are requested, see inputs section for more detail.
    - targets: another parameter to specify what to explain in the inputs, can be a specific class or a set of classes (for classification), or texts (for generation).
