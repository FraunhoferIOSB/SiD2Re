from typing import List

import numpy as np

from sid2re.driftgenerator.concept.nodes.base_node import BaseNode
from sid2re.driftgenerator.concept.utils import get_random_model
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class RandomFunctionNode(BaseNode):
    """Node representing a dependent feature."""

    def __init__(self, number_of_dependency_models: int, num_dependencies: int):
        """Initialize a RandomFunctionNode instance.

        Parameters
        ----------
        number_of_dependency_models : int
            Number of models to generate.
        num_dependencies : int
            Number of incoming dependencies.
        """
        super().__init__()
        self._collection = []
        self._tag_collection = []
        self._n_outputs = 1
        self._concept = False

        for _ in range(number_of_dependency_models):
            model_tag, model = get_random_model(num_dependencies, 1)
            self._collection.append(model)
            self._tag_collection.append(model_tag)

    def generate_data(self, time_stamps: NumberArray, inputs: List[NumberArray]) -> NumberArray:  # noqa:WPS210
        """Use to label inputs dependent on the time with this concept.

        As the concept might change over time, the time_stamps to the inputs have to be provided.

        Parameters
        ----------
        time_stamps : NumberArray
            Points in time of the given inputs. Shape: (n_samples)
        inputs : List[NumberArray]
            Inputs that are to be mapped to the targets. Shape: (feature_dim, n_samples)


        Returns
        -------
        NumberArray
            Target labels to the given inputs. Shape: (n_samples, target_dim)
        """
        init_flag = False
        labels: NumberArray = np.array([])
        input_values = np.dstack(inputs)[0]
        for (sample, stamp) in zip(input_values, time_stamps):
            output = np.zeros((1, self._n_outputs))
            coeffs = self.current_config(stamp)
            for (model, coeffizient, tag) in zip(self._collection, coeffs, self._tag_collection):
                if (tag in {'MLP', 'Decision Tree'}) and (len(model.predict(sample.reshape(1, -1)).shape) == 1):
                    model_prediction = model.predict(sample.reshape(1, -1))[:, None]
                else:
                    model_prediction = model.predict(sample.reshape(1, -1))
                output += coeffizient * model_prediction
            if init_flag:
                labels = np.concatenate((labels, output), axis=0)
            else:
                labels = output
                init_flag = True
        if labels.shape[1] == 1:
            labels = labels.flatten()
        return labels
