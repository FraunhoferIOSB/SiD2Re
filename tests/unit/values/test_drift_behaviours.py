import pytest
from sid2re.driftgenerator.concept.drift_behaviours import FaultySensorDriftBehaviour, GradualDriftBehaviour, \
    IncrementalDriftBehaviour, SuddenDriftBehaviour
from sid2re.driftgenerator.concept.drift_transition_functions import LinearTransitionFunction
import numpy as np

list_of_all_drift_behaviours = [FaultySensorDriftBehaviour, GradualDriftBehaviour,
                                IncrementalDriftBehaviour, SuddenDriftBehaviour]


@pytest.mark.parametrize("behaviour", list_of_all_drift_behaviours)
def test_non_reoccurring_drift_outer_behaviour(behaviour):
    coefficient = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    behaviour = behaviour(50, 10, coefficient_shift=coefficient.copy(), reoccurring=False)
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=10)
    assert (value == coefficient).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=70)
    assert (value == coefficient + 1).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()


def test_non_reoccurring_sudden_drift():
    behaviour = SuddenDriftBehaviour
    coefficient = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    behaviour = behaviour(50, 10, coefficient_shift=coefficient.copy(), reoccurring=False)
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=41)
    assert (value == coefficient + 1).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()


def test_non_reoccurring_faulty_sensor_drift():
    behaviour = FaultySensorDriftBehaviour
    coefficient = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    behaviour = behaviour(50, 10, coefficient_shift=coefficient.copy(), reoccurring=False)
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=41)
    assert (value == coefficient - 1).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=70)
    assert (value == coefficient + 1).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()


def test_non_reoccurring_gradual_drift():
    behaviour = GradualDriftBehaviour
    coefficient = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    behaviour = behaviour(50, 10, coefficient_shift=coefficient.copy(), reoccurring=False)
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=50)
    assert (value == coefficient).all() or (value == coefficient + 1).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()


def test_non_reoccurring_incremental_drift():
    behaviour = IncrementalDriftBehaviour
    coefficient = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    behaviour = behaviour(50, 10, coefficient_shift=coefficient.copy(), reoccurring=False)
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=50)
    assert (value == coefficient + 0.5).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()


@pytest.mark.parametrize("behaviour", list_of_all_drift_behaviours)
def test_reoccurring_drift_outer_behaviour(behaviour):
    coefficient = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    behaviour = behaviour(50, 10, coefficient_shift=coefficient.copy(), reoccurring=True)
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=10)
    assert (value == coefficient).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
    value = behaviour.modify_coeff(coefficient=coefficient, transition_behaviour=LinearTransitionFunction(),
                                   current_time=70)
    assert (value == coefficient).all()
    assert (coefficient == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
