import random
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


def get_random_model(number_of_features, number_of_outputs):
    selection = random.randint(1, 3)
    if selection == 1:
        # Multilayer Perceptron
        random_state = random.randint(1, 1000)
        X = (np.random.rand(20, number_of_features) - 0.5) * 10
        y = (np.random.rand(20, number_of_outputs) - 0.5) * 10
        if number_of_outputs == 1:
            y = y.flatten()
        return "MLP", MLPRegressor(random_state=random_state, hidden_layer_sizes=(random.randint(1, 100))).fit(X, y)

    if selection == 2:
        # Kernel Regression
        X = (np.random.rand(20, number_of_features) - 0.5) * 10
        y = (np.random.rand(20, number_of_outputs) - 0.5) * 10
        kernel_ridge = KernelRidge(
            kernel=random.choice(("poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"))).fit(X, y)
        return f"KernelRidge / {kernel_ridge.get_params()['kernel']}", kernel_ridge

    if selection == 3:
        # Tree Regression
        random_state = random.randint(1, 1000)
        X = (np.random.rand(20, number_of_features) - 0.5) * 10
        y = (np.random.rand(20, number_of_outputs) - 0.5) * 10
        return "Decision Tree", DecisionTreeRegressor(random_state=random_state).fit(X, y)
