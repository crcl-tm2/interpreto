---
icon: material/middleware-outline
---

# Base Classes

::: interpreto.concepts.ConceptEncoderExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - fit
        - verify_activations
        - top_k_inputs_for_concept
        - top_k_inputs_for_concept_from_activations
        - input_concept_attribution

::: interpreto.concepts.ConceptAutoEncoderExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - fit
        - encode_activations
        - decode_concepts
        - get_dictionary
        - top_k_inputs_for_concept
        - top_k_inputs_for_concept_from_activations
        - input_concept_attribution
        - concept_output_attribution
