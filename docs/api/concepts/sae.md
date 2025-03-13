---
icon: material/middleware-outline
---

# Sparse Autoencoders (SAEs)

::: interpreto.concepts.OvercompleteSAEClasses
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: interpreto.concepts.OvercompleteSAE
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - fit
        - encode_activations
        - decode_concepts
        - get_dictionary
        - input_concept_attribution
        - concept_output_attribution

## Loss Functions

These functions can be passed as the `criterion` argument in the `fit` method of the `OvercompleteSAE` class. `MSELoss` is the default loss function.

::: interpreto.concepts.methods.SAELoss
    handler: python
    options:
      show_root_heading: false
      show_source: false
      members:
        - __call__

::: interpreto.concepts.methods.MSELoss
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __call__

::: interpreto.concepts.methods.DeadNeuronsReanimationLoss
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __call__
