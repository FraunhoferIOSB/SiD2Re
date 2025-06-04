import random
from typing import Any, Tuple
from warnings import simplefilter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

simplefilter('ignore', category=ConvergenceWarning)


def get_random_model(number_of_features: int, number_of_outputs: int) -> Tuple[str, Any]:
    """Generate a random regression model.

    Parameters
    ----------
    number_of_features : int
        The number of input features.
    number_of_outputs : int
        The number of output targets.

    Returns
    -------
    Tuple[str, Any]
        A tuple containing the model label and the generated regression model.
    """
    selection = random.randint(1, 3)
    if selection == 1:
        # Multilayer Perceptron
        random_state = random.randint(1, 1000)
        features = (np.random.rand(20, number_of_features) - 0.5) * 10
        targets = (np.random.rand(20, number_of_outputs) - 0.5) * 10
        if number_of_outputs == 1:
            targets = targets.flatten()
        return 'MLP', MLPRegressor(random_state=random_state, hidden_layer_sizes=random.randint(1, 100)).fit(
            features,
            targets,
        )

    if selection == 2:
        # Kernel Regression
        features = (np.random.rand(20, number_of_features) - 0.5) * 10
        targets = (np.random.rand(20, number_of_outputs) - 0.5) * 10
        kernel_ridge = KernelRidge(
            kernel=random.choice(('poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine')),
        ).fit(features, targets)
        return f"KernelRidge / {kernel_ridge.get_params()['kernel']}", kernel_ridge

    # Standard Output: Tree Regression
    random_state = random.randint(1, 1000)
    features = (np.random.rand(20, number_of_features) - 0.5) * 10
    targets = (np.random.rand(20, number_of_outputs) - 0.5) * 10
    return 'Decision Tree', DecisionTreeRegressor(random_state=random_state).fit(features, targets)
