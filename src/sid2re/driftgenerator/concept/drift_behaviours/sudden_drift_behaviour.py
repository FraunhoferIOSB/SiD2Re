from sid2re.driftgenerator.concept.drift_behaviours._base_drift_behaviour import _BaseDriftBehaviour
from sid2re.driftgenerator.concept.drift_transition_functions import _BaseTransitionFunction
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class SuddenDriftBehaviour(_BaseDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by a sudden drift.

    A sudden drift instantly implements a new concept.
    """
    def _compute_coeff_delta(
        self,
        coefficient: NumberArray,
        transition_behaviour: _BaseTransitionFunction,
        current_time: float
    ) -> NumberArray:
        """Compute the change in coefficients during sudden drift.

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
        coefficient_delta = self.coefficient_shift
        return coefficient_delta

    @property
    def _drift_name(self) -> str:
        """The name of the drift behavior.

        Returns
        -------
        str
            The name of the drift behavior: "sudden".
        """
        return "sudden"
