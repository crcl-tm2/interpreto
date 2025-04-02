---
icon: material/bookshelf
---

# Concepts Dictionary Metrics

Concept dictionary metrics evaluate the concept space via its dictionary.
They can either be compute on a single dictionary or compare two or more dictionaries.

```python
from interpreto.concepts.metrics import MetricClass

metric = MetricClass(concept_explainer1, concept_explainer2, ...)
score = metric.compute()
```



## Stability

::: interpreto.concepts.metrics.Stability
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
