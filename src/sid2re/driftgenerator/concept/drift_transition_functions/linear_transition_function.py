from sid2re.driftgenerator.concept.drift_transition_functions.base_transition_function import BaseTransitionFunction


class LinearTransitionFunction(BaseTransitionFunction):
    """Transition function used by incremental drifts. This is a linear transition."""

    @classmethod
    def _transition_coefficient(cls, current_time: float, shift_centre: float, shift_radius: float) -> float:
        """Compute the transition coefficient using linear interpolation.

        Parameters
        ----------
        current_time : float
            The current time in the system.
        shift_centre : float
            The center of the shift.
        shift_radius : float
            The radius around the shift center.

        Returns
        -------
        float
            The computed transition coefficient.
        """
        drift_begin = shift_centre - shift_radius
        drift_end = shift_centre + shift_radius
        drift_interval = drift_end - drift_begin
        passed_interval = current_time - drift_begin

        return passed_interval / drift_interval
