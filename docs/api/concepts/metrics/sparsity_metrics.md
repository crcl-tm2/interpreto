---
icon: material/poll
---

# Concepts Sparsity Metrics

Concept sparsity metrics evaluate the sparsity of the concept-space activations.
They take in the `concept_explainer` and the `latent_activations`, compute the `concept_activations` and then compute the sparsity of the `concept_activations`.

```python
from interpreto.concepts.metrics import MetricClass

metric = MetricClass(concept_explainer)
score = metric.compute(activations)
```



## Sparsity

::: interpreto.concepts.metrics.Sparsity
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true



## Sparsity Ratio

::: interpreto.concepts.metrics.SparsityRatio
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
