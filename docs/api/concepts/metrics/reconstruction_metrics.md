---
icon: material/vector-difference
---

# Concepts Reconstruction Metrics

Concept reconstruction error measures the faithfulness of the concept space with respect to the explained model latent space. To do so, these metrics compute the distance between initial activations and the reconstructed activations.

```python
from interpreto.concepts.metrics import MetricClass

metric = MetricClass(concept_explainer)
score = metric.compute(activations)
```



## MSE

::: interpreto.concepts.metrics.MSE
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true



## FID

::: interpreto.concepts.metrics.FID
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true



## Custom

To tune the reconstruction error to your need, you can specify a `reconstruction_space` and a `distance_function`, note that the `distance_function` should follow the `interpreto.commons.DistanceFunctionProtocol` protocol.

::: interpreto.concepts.metrics.ReconstructionError
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: interpreto.concepts.metrics.ReconstructionSpaces
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: interpreto.commons.DistanceFunctions
    handler: python
    options:
      show_root_heading: true
      show_source: true
