from sid2re.driftgenerator.concept.drift_behaviours.base_drift_behaviour import (
    BaseDriftBehaviour,
)
from sid2re.driftgenerator.concept.drift_transition_functions import (
    BaseTransitionFunction,
)
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class SuddenDriftBehaviour(BaseDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by a sudden drift.

    A sudden drift instantly implements a new concept.
    """

    def __init__(
        self,
        drift_time: float,
        drift_radius: float,
        coefficient_shift: NumberArray,
        reoccurring: bool = False,
    ) -> None:
        """
        Initialize the drift behavior.

        Parameters
        ----------
        drift_time : float
            The time at which the drift occurs.
        drift_radius : float
            The radius around the drift time within which the drift effect is active.
        coefficient_shift : NumberArray
            The amount by which coefficients are shifted during the drift.
        reoccurring : bool
            Indicates if the drift occurs repeatedly.
        """
        super().__init__(
            drift_time=drift_time,
            drift_radius=drift_radius,
            coefficient_shift=coefficient_shift,
            reoccurring=reoccurring,
        )

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


class ReoccuringSuddenDriftBehaviour(SuddenDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by a sudden drift.

    A sudden drift instantly implements a new concept.

    This drift is reoccuring, so it will go trough a phase transition A -> B -> A.
    """

    def __init__(
        self,
        drift_time: float,
        drift_radius: float,
        coefficient_shift: NumberArray,
        reoccurring: bool = True,
    ) -> None:
        """
        Initialize the drift behavior.

        Parameters
        ----------
        drift_time : float
            The time at which the drift occurs.
        drift_radius : float
            The radius around the drift time within which the drift effect is active.
        coefficient_shift : NumberArray
            The amount by which coefficients are shifted during the drift.
        reoccurring : bool
            Indicates if the drift occurs repeatedly.
        """
        super().__init__(
            drift_time=drift_time,
            drift_radius=drift_radius,
            coefficient_shift=coefficient_shift,
            reoccurring=reoccurring,
        )
