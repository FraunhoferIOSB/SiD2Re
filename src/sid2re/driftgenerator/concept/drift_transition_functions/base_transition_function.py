from abc import ABC, abstractmethod


class BaseTransitionFunction(ABC):
    """Abstract base class for any transition functions in sid2re."""

    @classmethod
    def transition_coefficient(cls, current_time: float, shift_centre: float, shift_radius: float) -> float:
        """Compute the transition coefficient.

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
            The transition coefficient, ranging from 0 to 1.

        Raises
        ------
        ValueError
            If shift_radius is negative or zero.

        """
        if current_time < shift_centre - shift_radius:
            return 0
        if current_time > shift_centre + shift_radius:
            return 1
        if shift_radius <= 0:
            raise ValueError('Shift Radius cannot be negative or zero')
        return cls._transition_coefficient(current_time, shift_centre, shift_radius)

    @classmethod
    @abstractmethod
    def _transition_coefficient(cls, current_time: float, shift_centre: float, shift_radius: float) -> float:
        """Abstract method to compute the transition coefficient.

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
