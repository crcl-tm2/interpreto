import os

import matplotlib.colors as mcolors
import torch
from matplotlib import pyplot as plt

from interpreto.attributions.base import AttributionOutput
from interpreto.visualizations.attributions.classification_highlight import (
    GenerationAttributionVisualization,
    MultiClassAttributionVisualization,
    SingleClassAttributionVisualization,
)
from interpreto.visualizations.concepts.concepts_highlight import (
    ConceptHighlightVisualization,
)


def test_attribution_monoclass():
    # attributions (1 classe)
    inputs_sentences = [["A", "B", "C", "one", "two", "three"], ["do", "re", "mi"]]
    nb_classes = 1

    def gen_attributions_outputs(sentences):
        for sentence in sentences:
            attributions = torch.rand(len(sentence), nb_classes)  # (l, c)
            yield AttributionOutput(elements=sentence, attributions=attributions)

    single_class_classification_output = list(gen_attributions_outputs(inputs_sentences))
    viz = SingleClassAttributionVisualization(
        attribution_output_list=single_class_classification_output,
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_attribution_monoclass.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)


def test_attribution_multiclass():
    # attributions (2 classes)
    inputs_sentences = [["A", "B", "C", "one", "two", "three"], ["do", "re", "mi"]]
    nb_classes = 2
    inputs_sentences = [["A", "B", "C", "one", "two", "three"], ["do", "re", "mi"]]

    def gen_attributions_outputs(sentences):
        for sentence in sentences:
            attributions = torch.rand(len(sentence), nb_classes)  # (l, c)
            yield AttributionOutput(elements=sentence, attributions=attributions)

    multi_class_classification_output = list(gen_attributions_outputs(inputs_sentences))

    viz = MultiClassAttributionVisualization(
        attribution_output_list=multi_class_classification_output,
        class_colors=[mcolors.to_rgb("green"), mcolors.to_rgb("blue")],
        class_names=["class1", "class2"],
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_attribution_multiclass.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)


def test_attribution_generation():
    inputs_sentence = ["A", "B", "C", "one", "two", "three"]
    outputs_sentence = ["do", "re", "mi"]

    def make_attributions_outputs(inputs, outputs):
        attributions = torch.rand(len(inputs) + len(outputs), len(outputs))  # (l, l_g)
        return AttributionOutput(elements=inputs + outputs, attributions=attributions)

    generation_output = make_attributions_outputs(inputs_sentence, outputs_sentence)

    viz = GenerationAttributionVisualization(attribution_output=generation_output)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_attribution_generation.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)


def test_concepts():
    # Concepts: 9 concepts (with inputs attributions for each output word)

    inputs_sentence = ["A", "B", "C", "one", "two", "three"]
    outputs_sentence = ["do", "re", "mi"]
    nb_concepts = 9

    def make_attributions_outputs(inputs, outputs):
        attributions = torch.rand(len(inputs) + len(outputs), len(outputs), nb_concepts)  # (l, l_g, c)
        return AttributionOutput(elements=inputs + outputs, attributions=attributions)

    generation_output = make_attributions_outputs(inputs_sentence, outputs_sentence)
    colors_set1 = plt.get_cmap("Set1").colors

    viz = ConceptHighlightVisualization(generation_output, concepts_colors=colors_set1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_concepts.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)
