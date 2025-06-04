from abc import ABC, abstractmethod
from typing import List

from sid2re.driftgenerator.concept.drift_transition_functions import BaseTransitionFunction
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class BaseDriftBehaviour(ABC):
    """Abstract base class defining the behavior of drift in a system."""

    def __init__(
        self,
        drift_time: float,
        drift_radius: float,
        coefficient_shift: NumberArray,
        reoccurring: bool,
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
        self.drift_time = drift_time
        self.drift_radius = drift_radius
        self.coefficient_shift = coefficient_shift
        self.drift_begin = drift_time - drift_radius
        self.drift_end = drift_time + drift_radius
        self.reoccurring = reoccurring
        if self.reoccurring:
            self.to_drift = self.__class__(drift_time - (drift_radius / 2), drift_radius / 2, coefficient_shift, False)
            self.from_drift = self.__class__(
                drift_time + (drift_radius / 2), drift_radius / 2, -coefficient_shift,
                False,
            )

    @property
    def drift_information(self) -> List:
        """Fetch information about the drift as a list [Time, Radius, Coefficient,Name].

        Returns
        -------
        List
        """
        return [self.drift_time, self.drift_radius, self.coefficient_shift, self._drift_name]

    def modify_coeff(
        self,
        coefficient: NumberArray,
        transition_behaviour: BaseTransitionFunction,
        current_time: float,
    ) -> NumberArray:
        """
        Modify the coefficients based on the drift behavior.

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
            The modified coefficients.
        """
        coefficient = coefficient.copy()  # prevent manipulation of the initial input coefficient
        if self.reoccurring:
            coefficient = self.to_drift.modify_coeff(coefficient, transition_behaviour, current_time)
            coefficient = self.from_drift.modify_coeff(coefficient, transition_behaviour, current_time)
        else:
            if self.drift_begin < current_time <= self.drift_end:
                coefficient = coefficient + self._compute_coeff_delta(coefficient, transition_behaviour, current_time)
            elif current_time > self.drift_end:
                coefficient = coefficient + self.coefficient_shift

        return coefficient

    @abstractmethod
    def _compute_coeff_delta(  # noqa: DOC101,DOC109,DOC103,DOC201,DOC203
        self,
        coefficient: NumberArray,
        transition_behaviour: BaseTransitionFunction,
        current_time: float,
    ) -> NumberArray:
        """
        Abstract method to compute the change in coefficients during drift.

        Parameters
        ----------
        coefficient : NumberArray
            The coefficients to be updated.
        transition_behaviour : BaseTransitionFunction
            The transition behavior.
        current_time : float
            The current time in the simulation.

        Returns
        -------
        NumberArray
            The updated coefficients.
        """

    @property
    @abstractmethod
    def _drift_name(self) -> str:
        """
        Abstract property representing the name of the drift behavior.

        Returns
        -------
        str
            The name of the drift behavior.
        """
