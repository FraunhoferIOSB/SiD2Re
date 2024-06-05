from sid2re.driftgenerator.concept.drift_behaviours._base_drift_behaviour import _BaseDriftBehaviour
from sid2re.driftgenerator.concept.drift_transition_functions import _BaseTransitionFunction
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class IncrementalDriftBehaviour(_BaseDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by an incremental drift.

    An incremental drift represents a continuous transfer from one concept to another. The coefficients are therefore
    interpolated between the start and the end state.
    """

    def _compute_coeff_delta(
        self,
        coefficient: NumberArray,
        transition_behaviour: _BaseTransitionFunction,
        current_time: float
    ) -> NumberArray:
        """
        Compute the change in coefficients during incremental drift.

        Parameters
        ----------
        coefficient : NumberArray
            The coefficients to be modified.
        transition_behaviour : _BaseTransitionFunction
            The behavior of transition.
        current_time : float
            The current time in the system.

        Returns
        -------
        NumberArray
            The computed change in coefficients.
        """
        coefficient_delta = transition_behaviour.transition_coefficient(
            current_time, self.drift_time,
            self.drift_radius
        ) * self.coefficient_shift
        return coefficient_delta

    @property
    def _drift_name(self) -> str:
        """
        The name of the drift behavior.

        Returns
        -------
        str
            The name of the drift behavior: "incremental".
        """
        return "incremental"
