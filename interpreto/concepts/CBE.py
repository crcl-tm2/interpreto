from base import AbstractConceptExplainer


class ConceptBottleneckExplainer(AbstractConceptExplainer):
    """
    Implementation of a concept explainer based on the Concept Bottleneck model.
    """

    def fit(self, inputs, split):
        """
        Method for explaining a concept.

        :param inputs: The inputs to be explained.
        :return: the concept encoder and the concept decoder.
        """
        #self.split = split
        #self.encoder = ...
        #self.decoder = ...
        pass


    def concept_output_attribution(self, concepts, attribution_method):
        """
        Computes the attribution of each concept to a given example.

        :param concepts: The concepts to analyze.
        :param attribution_method: The method to use for attribution analysis.
        :return: A list of attribution scores for each concept.
        """
        pass