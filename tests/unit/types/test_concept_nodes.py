from sid2re.driftgenerator.concept.nodes import RandomConceptFunctionNode, RandomFunctionNode, RootFeatureNode
from sid2re.driftgenerator.concept.nodes.root_distributions import UniformDistribution
from sid2re.driftgenerator.concept.drift_transition_functions import LinearTransitionFunction
from sid2re.driftgenerator.concept.drift_behaviours import SuddenDriftBehaviour, FaultySensorDriftBehaviour, \
    GradualDriftBehaviour, IncrementalDriftBehaviour
import pytest
import numpy as np


@pytest.mark.parametrize("n_models", [5, 10])
@pytest.mark.parametrize("n_dependencies", [1, 2])
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_concept_node_without_drift(n_models, n_dependencies, n_outputs):
    node = RandomConceptFunctionNode(number_of_dependency_models=n_models, num_dependencies=n_dependencies,
                                     n_outputs=n_outputs)
    data = node.generate_data(time_stamps=np.arange(100), inputs=np.random.normal(3, 2.5, size=(n_dependencies, 100)))
    assert data.shape[0] == 100
    if n_outputs != 1:
        assert data.shape[1] == n_outputs


@pytest.mark.parametrize("n_models", [5, 10])
@pytest.mark.parametrize("n_dependencies", [1, 2])
def test_function_node_without_drift(n_models, n_dependencies):
    node = RandomFunctionNode(number_of_dependency_models=n_models, num_dependencies=n_dependencies)
    data = node.generate_data(time_stamps=np.arange(100), inputs=np.random.normal(3, 2.5, size=(n_dependencies, 100)))
    assert data.shape[0] == 100


def test_root_node_without_drift():
    node = RootFeatureNode(minimum=-2, maximum=8)
    node.set_distro(UniformDistribution())
    data = node.generate_data(time_stamps=np.arange(100))
    assert np.amin(data) > -2
    assert np.amax(data) < 8
    assert data.shape[0] == 100


@pytest.mark.parametrize("n_dependencies", [1, 2])
@pytest.mark.parametrize("n_outputs", [1, 2])
@pytest.mark.parametrize("drift_behaviour", [SuddenDriftBehaviour, FaultySensorDriftBehaviour, GradualDriftBehaviour,
                                             IncrementalDriftBehaviour])
@pytest.mark.parametrize("reoccurring", [True, False])
def test_concept_node_with_drift(n_dependencies, n_outputs, drift_behaviour, reoccurring):
    node = RandomConceptFunctionNode(number_of_dependency_models=5, num_dependencies=n_dependencies,
                                     n_outputs=n_outputs)
    drifts = [drift_behaviour(drift_time=50, drift_radius=10, coefficient_shift=np.array([1, 1, 1, 1, 1]),
                              reoccurring=reoccurring)]
    node.set_drift(drifts)
    data = node.generate_data(time_stamps=np.arange(100), inputs=np.random.normal(3, 2.5, size=(n_dependencies, 100)))
    assert data.shape[0] == 100
    if n_outputs != 1:
        assert data.shape[1] == n_outputs


@pytest.mark.parametrize("n_dependencies", [1, 2])
@pytest.mark.parametrize("drift_behaviour", [SuddenDriftBehaviour, FaultySensorDriftBehaviour, GradualDriftBehaviour,
                                             IncrementalDriftBehaviour])
@pytest.mark.parametrize("reoccurring", [True, False])
def test_function_node_with_drift(n_dependencies, drift_behaviour, reoccurring):
    node = RandomFunctionNode(number_of_dependency_models=5, num_dependencies=n_dependencies)
    node.set_transition(LinearTransitionFunction())
    drifts = [
        drift_behaviour(drift_time=50, drift_radius=10, coefficient_shift=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                        reoccurring=reoccurring)]
    node.set_drift(drifts)
    data = node.generate_data(time_stamps=np.arange(100), inputs=np.random.normal(3, 2.5, size=(n_dependencies, 100)))
    assert data.shape[0] == 100
