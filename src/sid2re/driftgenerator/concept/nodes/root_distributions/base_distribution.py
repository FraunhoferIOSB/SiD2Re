from abc import ABC, abstractmethod

from sid2re.driftgenerator.utils.type_aliases import NumberArray


class BaseDistribution(ABC):
    """Abstract base class defining the distribution behavior."""

    @classmethod
    def generate(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:  WPS123
        """Generate a value based on the distribution behavior.

        Parameters
        ----------
        w : NumberArray
            The weights for distribution generation.
        timestamp : float
            The timestamp for distribution generation.
        minimum : float
            The minimum value for distribution generation.
        maximum : float
            The maximum value for distribution generation.

        Returns
        -------
        float
            The generated value.
        """
        if not w.any():
            return float(0)
        return cls._generate(w, timestamp, minimum, maximum)

    @classmethod
    @abstractmethod
    def _generate(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:WPS123,WPS111
        """Abstract method to generate a value based on the distribution behavior.

        Parameters
        ----------
        w : NumberArray
            The weights for distribution generation.
        timestamp : float
            The timestamp for distribution generation.
        minimum : float
            The minimum value for distribution generation.
        maximum : float
            The maximum value for distribution generation.

        Returns
        -------
        float
            The generated value.
        """
