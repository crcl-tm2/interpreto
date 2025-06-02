"""
Base classes for attributions visualizations
"""

from __future__ import annotations

import json

from interpreto.attributions.base import AttributionOutput
from interpreto.visualizations.base import AttributionVisualization, tensor_to_list


class SingleClassAttributionVisualization(AttributionVisualization):
    """
    Class for attributions visualization for classification models (monoclass)
    """

    def __init__(
        self,
        attribution_output: AttributionOutput,
        positive_color: str = "#ff0000",
        negative_color: str = "#0000ff",
        normalize: bool = True,
        highlight_border: bool = False,
        css: str = None,
    ):
        """
        Create a mono class attribution visualization

        Args:
            attribution_output: AttributionOutput: The attribution method outputs
            positive_color (str, optional): A hexadecimal color code in RGB format for positive activations. The default color is red (#ff0000)
            negative_color (str, optional): A hexadecimal color code in RGB format for negative activations. The default color is blue (#0000ff)
            normalize (bool, optional): Whether to normalize the attributions. If False, then the attributions values range will be assumed to be [-1, 1]. Defaults to True
            highlight_border (bool, optional): Whether to highlight the border of the words. Defaults to False
            css: (str, optional): A custom css. Defaults to None
        """
        super().__init__()
        inputs_sentence = attribution_output.elements
        # format of attributions for 1 class attribution:
        # nb_sentences * (1, nb_words, 1) with the first dimension beeing the number
        # of generated outputs (here set to 1 because no generation)
        # and the last the number of classes (here set to 1 because only one class)
        inputs_attribution = attribution_output.attributions.unsqueeze(0).unsqueeze(-1)

        # compute the min and max values for the attributions to be used for normalization
        if normalize:
            min_value = inputs_attribution.min()
            max_value = inputs_attribution.max()
        else:
            min_value = -1.0
            max_value = 1.0
        assert min_value <= max_value, "The min value should be less than the max value"

        self.highlight_border = highlight_border
        self.custom_css = css
        self.data = self.adapt_data(
            input_words=inputs_sentence,
            input_attributions=inputs_attribution,
            output_words=None,
            output_attributions=None,
            classes_descriptions=self.make_classes_descriptions(
                positive_color, negative_color, min_value=min_value, max_value=max_value
            ),
        )

    def make_classes_descriptions(
        self,
        positive_color: str,
        negative_color: str,
        name: str = "None",
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        """
        Create a structure describing the classes

        Args:
            color (Tuple): A color for the class
            name (str, optional): The name of the class. Defaults to "None".
            min_value (float, optional): The minimum value for the attributions. Defaults to -1.0.
            max_value (float, optional): The maximum value for the attributions. Defaults to 1.0.

        Returns:
            dict: A dictionary describing the class
        """
        return [
            {
                "name": f"class #{name}",
                "description": f"This is the description of class #{name}",
                "positive_color": positive_color,
                "negative_color": negative_color,
                "min": min_value,
                "max": max_value,
            }
        ]

    def build_html(self):
        """
        Build the html for the visualization
        """
        json_data = json.dumps(self.data, default=tensor_to_list, indent=2)
        html = self.build_html_header()
        html += f"<h3>Inputs</h3><div id='{self.unique_id_inputs}'></div>\n"
        html += f"""
        <script>
            var viz = new DataVisualizationAttribution(null, '{self.unique_id_inputs}', null, '{self.highlight_border}', {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html


class MultiClassAttributionVisualization(AttributionVisualization):
    """
    Class for attributions visualization for classification models (multiclass)
    """

    def __init__(
        self,
        attribution_output: AttributionOutput,
        positive_color: str = "#ff0000",
        negative_color: str = "#0000ff",
        class_names: list[str] = None,
        normalize: bool = True,
        highlight_border: bool = False,
        css: str = None,
    ):
        """
        Create a multi class attribution visualization

        Args:
            attribution_output: AttributionOutput: The attribution method output
            positive_color (str, optional): A hexadecimal color code in RGB format for positive activations. The default color is red (#ff0000)
            negative_color (str, optional): A hexadecimal color code in RGB format for negative activations. The default color is blue (#0000ff)
            class_names (List[str], optional): A list of names for each class. Defaults to None
            normalize (bool, optional): Whether to normalize the attributions. If False, then the attributions values range will be assumed to be [-1, 1]. Defaults to True
            highlight_border (bool, optional): Whether to highlight the border of the words. Defaults to False
            css: (str, optional): A custom css. Defaults to None
        """
        super().__init__()

        inputs_sentence = attribution_output.elements

        # format of attributions for multi class attribution:
        # nb_sentences * (1, nb_words, nb_classes) with the first dimension beeing the number
        # of generated outputs (here set to 1 because no generation)
        # inputs_attributions = [output.attributions.T.unsqueeze(0) for output in attribution_output_list]
        nb_classes, _ = attribution_output.attributions.shape
        inputs_attributions = attribution_output.attributions.T.unsqueeze(0)
        if class_names is None:
            class_names = [f"class #{c}" for c in range(nb_classes)]

        # compute the min and max values for the attributions to be used for normalization
        if normalize:
            min_values = attribution_output.attributions.min(axis=1).values
            max_values = attribution_output.attributions.max(axis=1).values
        else:
            min_values = [-1.0] * nb_classes
            max_values = [1.0] * nb_classes

        self.highlight_border = highlight_border
        self.custom_css = css
        self.data = self.adapt_data(
            input_words=inputs_sentence,
            input_attributions=inputs_attributions,
            output_words=None,
            output_attributions=None,
            classes_descriptions=self.make_classes_descriptions(
                positive_color, negative_color, class_names, min_values=min_values, max_values=max_values
            ),
        )

    def make_classes_descriptions(
        self,
        positive_color: str,
        negative_color: str,
        class_names: list[str],
        min_values: list[float],
        max_values: list[float],
    ):
        """
        Create a structure describing the classes

        Args:
            positive_color (str): A hexadecimal color code in RGB format for positive activations
            negative_color (str): A hexadecimal color code in RGB format for negative activations
            class_names (List[str]): A list of names for each class
            min_value (List): The minimum values for the attributions
            max_value (List): The maximum values for the attributions

        Returns:
            dict: A dictionary describing the classes
        """
        return [
            {
                "name": f"{name}",
                "description": f"This is the description of class #{name}",
                "positive_color": positive_color,
                "negative_color": negative_color,
                "min": min_value,
                "max": max_value,
            }
            for name, min_value, max_value in zip(class_names, min_values, max_values, strict=False)
        ]

    def build_html(self):
        """
        Build the html for the visualization
        """
        json_data = json.dumps(self.data, default=tensor_to_list, indent=2)
        html = self.build_html_header()
        html += f"<h3>Classes</h3><div class='line-style'><div id='{self.unique_id_classes}'></div></div>\n"
        html += f"<h3>Inputs</h3><div id='{self.unique_id_inputs}'></div>\n"
        html += f"""
        <script>
            var viz = new DataVisualizationAttribution('{self.unique_id_classes}', '{self.unique_id_inputs}', null, '{self.highlight_border}', {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html


class GenerationAttributionVisualization(AttributionVisualization):
    """
    Class for attributions visualization when using a generative model
    """

    def __init__(
        self,
        attribution_output: AttributionOutput,
        positive_color: str = "#ff0000",
        negative_color: str = "#0000ff",
        normalize: bool = True,
        highlight_border: bool = False,
        css: str = None,
    ):
        """
        Initialize the visualization

        Args:
            attribution_output: AttributionOutput: The attribution outputs to visualize
            positive_color (str, optional): A hexadecimal color code in RGB format for positive activations. The default color is red (#ff0000)
            negative_color (str, optional): A hexadecimal color code in RGB format for negative activations. The default color is blue (#0000ff)
            normalize (bool, optional): Whether to normalize the attributions. If False, then the attributions values range will be assumed to be [-1, 1]. Defaults to True
            highlight_border (bool, optional): Whether to highlight the border of the words. Defaults to False
            css: (str, optional): A custom css. Defaults to None
        """
        super().__init__()
        nb_outputs, nb_inputs_outputs = attribution_output.attributions.shape
        nb_inputs = nb_inputs_outputs - nb_outputs
        assert nb_inputs_outputs == len(attribution_output.elements), (
            f"The attribution shape ({nb_inputs_outputs}) does not match the number of elements ({len(attribution_output.elements)})"
        )

        # reformat attribution_output to match the expected format for the js visualization
        inputs_words = attribution_output.elements[:nb_inputs]
        outputs_words = attribution_output.elements[nb_inputs:]

        # split the attributions into input_attributions and output_attributions
        # attribution shape is (nb_outputs, nb_inputs + nb_outputs)
        # js expects inputs attributions of shape (nb_outputs, nb_inputs, 1)
        # and outputs attributions of shape (nb_outputs, nb_outputs, 1)
        inputs_attributions = attribution_output.attributions[:, :nb_inputs].unsqueeze(-1)
        assert inputs_attributions.shape == (nb_outputs, nb_inputs, 1), (
            f"The inputs attributions shape ({inputs_attributions.shape}) \
            does not match the expected shape ({nb_outputs}, {nb_inputs}, 1)"
        )

        outputs_attributions = attribution_output.attributions[:, nb_inputs:].unsqueeze(-1)
        assert outputs_attributions.shape == (nb_outputs, nb_outputs, 1), (
            f"The outputs attributions shape ({outputs_attributions.shape}) \
            does not match the expected shape ({nb_outputs}, {nb_outputs}, 1)"
        )

        # compute the min and max values for the attributions to be used for normalization
        if normalize:
            min_value = attribution_output.attributions.min()
            max_value = attribution_output.attributions.max()
            assert min_value <= max_value, "The min value should be less than the max value"
        else:
            min_value = -1.0
            max_value = 1.0

        self.highlight_border = highlight_border
        self.custom_css = css
        self.data = self.adapt_data(
            input_words=inputs_words,
            input_attributions=inputs_attributions,
            output_words=outputs_words,
            output_attributions=outputs_attributions,
            classes_descriptions=self.make_classes_descriptions(
                positive_color=positive_color, negative_color=negative_color, min_value=min_value, max_value=max_value
            ),
        )

    def make_classes_descriptions(
        self,
        positive_color: str,
        negative_color: str,
        name: str = "None",
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        """
        Create a structure describing the classes

        Args:
            positive_color (str): A hexadecimal color code in RGB format for positive activations
            negative_color (str): A hexadecimal color code in RGB format for negative activations
            name (str, optional): The name of the class. Defaults to "None".
            min_value (float): The minimum value for the attributions. Defaults to -1.0.
            max_value (float): The maximum value for the attributions. Defaults to 1.0.

        Returns:
            dict: A dictionary describing the class
        """
        return [
            {
                "name": f"class #{name}",
                "description": f"This is the description of class #{name}",
                "positive_color": positive_color,
                "negative_color": negative_color,
                "min": min_value,
                "max": max_value,
            }
        ]

    def build_html(self):
        """
        Build the HTML visualization
        """
        json_data = json.dumps(self.data, default=tensor_to_list, indent=2)
        html = self.build_html_header()
        html += f"<h3>Inputs</h3><div id='{self.unique_id_inputs}'></div>\n"
        html += f"<h3>Outputs</h3><div class='line-style'><div id='{self.unique_id_outputs}'></div></div>\n"
        html += f"""
        <script>
            var viz = new DataVisualizationAttribution(null, '{self.unique_id_inputs}', '{self.unique_id_outputs}', '{self.highlight_border}', {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html
