from sid2re.driftgenerator.concept.drift_transition_functions import LinearTransitionFunction
import pytest


@pytest.mark.parametrize("transition_function", [LinearTransitionFunction])
def test_transition_function_compliance(transition_function):
    value = transition_function.transition_coefficient(current_time=0, shift_centre=10, shift_radius=5)
    assert value == 0
    value = transition_function.transition_coefficient(current_time=20, shift_centre=10, shift_radius=5)
    assert value == 1
    value = transition_function.transition_coefficient(current_time=10, shift_centre=10, shift_radius=5)
    assert 0 < value < 1
    with pytest.raises(ValueError) as excinfo:
        _ = transition_function.transition_coefficient(current_time=10, shift_centre=10, shift_radius=0)
    assert len(str(excinfo.value)) > 0


def test_linear_transition_function():
    transition_function = LinearTransitionFunction()
    value = transition_function.transition_coefficient(current_time=2, shift_centre=5, shift_radius=5)
    assert value == 0.2
    value = transition_function.transition_coefficient(current_time=6, shift_centre=5, shift_radius=5)
    assert value == 0.6
