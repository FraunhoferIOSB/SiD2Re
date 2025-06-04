from sid2re.driftgenerator.concept.drift_behaviours.base_drift_behaviour import BaseDriftBehaviour
from sid2re.driftgenerator.concept.drift_transition_functions import BaseTransitionFunction
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class SuddenDriftBehaviour(BaseDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by a sudden drift.

    A sudden drift instantly implements a new concept.
    """

    def _compute_coeff_delta(
        self,
        coefficient: NumberArray,
        transition_behaviour: BaseTransitionFunction,
        current_time: float,
    ) -> NumberArray:
        """Compute the change in coefficients during sudden drift.

        Parameters
        ----------
        coefficient : NumberArray
            The coefficients to be modified.
        transition_behaviour : BaseTransitionFunction
            The behavior of transition.
        current_time : float
            The current time in the system.

        Returns
        -------
        NumberArray
            The computed change in coefficients.
        """
        return self.coefficient_shift

    @property
    def _drift_name(self) -> str:
        """The name of the drift behavior.

        Returns
        -------
        str
            The name of the drift behavior: "sudden".
        """
        return 'sudden'
