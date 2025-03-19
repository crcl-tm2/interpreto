
#API General

```
model_with_split_points = ModelWithSplitPoints(model, tokenizer, split_points)
concept_explainer = TopKSAEExplainer(model_with_split_points)

activations = model_with_split_points.get_activations(dataset, split_point)
concept_explainer.fit(activations)


integrated_gradient_kwargs = {"step": 10, "baseline": "[MASK]"}
concept_input_attribution = concept_explainer.input_concept_attribution(attribution_method=IntegratedGradient, examples, **integrated_gradient_kwargs)

TopKtokens_kwargs = {todo}
concept_interpretation = concept_explainer.interpret(interpretation_method=TopKtokens, examples, **TopKtokens_kwargs)

```



#Custom API

```
model_with_split_points = ModelWithSplitPoints(model, tokenizer, split_points)
concept_model = SAEExtraction(concept_model=OvercompleteSAEClasses.TopKSAE)
activations = model_with_split_points.get_activations(dataset, split_point)

concept_model.fit(activations)

concept_explainer = ConceptAutoEncoderExplainer(model_with_split_points, concept_model)

integrated_gradient_kwargs = {"step": 10, "baseline": "[MASK]"}
concept_input_attribution = concept_explainer.input_concept_attribution(attribution_method=IntegratedGradient, examples, **integrated_gradient_kwargs)

TopKtokens_kwargs = {todo}
concept_interpretation = concept_explainer.interpret(interpretation_method=TopKtokens, examples, **TopKtokens_kwargs)

```

