from sid2re.driftgenerator.concept.drift_behaviours.base_drift_behaviour import (
    BaseDriftBehaviour,
)
from sid2re.driftgenerator.concept.drift_transition_functions import (
    BaseTransitionFunction,
)
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class FaultySensorDriftBehaviour(BaseDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by a faulty sensor.

    A faulty sensor shuts down any information gained by the sensor. This is expressed by setting the coefficient to
    zero so that information is deleted.
    The transformation is therefore the inverse of the current coefficients.
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
        """
        Compute the change in coefficients during drift caused by a faulty sensor.

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
        return -coefficient

    @property
    def _drift_name(self) -> str:
        """
        The name of the drift behavior.

        Returns
        -------
        str
            The name of the drift behavior: "faulty_sensor".
        """
        return 'faulty_sensor'


class ReoccuringFaultySensorDriftBehaviour(FaultySensorDriftBehaviour):
    """Specify the transformation of the concept coefficients caused by a faulty sensor.

    A faulty sensor shuts down any information gained by the sensor. This is expressed by setting the coefficient to
    zero so that information is deleted.
    The transformation is therefore the inverse of the current coefficients.

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
