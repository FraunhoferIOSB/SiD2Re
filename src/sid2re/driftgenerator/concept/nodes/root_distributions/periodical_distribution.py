import math

import numpy as np

from sid2re.driftgenerator.concept.nodes.root_distributions._base_distribution import _BaseDistribution
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class PeriodicalDistribution(_BaseDistribution):
    """Value generating distribution used in root features.

    This distributions generation follows a time dependent cycle.
    """

    @classmethod
    def _generate(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:  WPS123
        """
        Generate a value based on a time-dependent cycle.

        Parameters
        ----------
        w : NumberArray
            Behaviour defining vector of values between 0 and 1.
            w[0]: share of the interval used for shifting the function along the time axis.
            w[1]: defining periodical interval between 5 and 50.
            w[2]: share of the [feature_min, feature_max] interval, used for value axis shift.
            w[3]: proportional to pitch of the linear function.
            w[4]: used to determine whether the pitch of the linear function is positive or negative.
        timestamp : float
            Point in time for which the function value of the transformed linear function is asked.
        minimum : float
            Minimal function value that is allowed to be generated.
        maximum : float
            Maximal function value that is allowed to be generated.

        Returns
        -------
        float
            The generated value.
        """
        selection = math.floor(3 * w[0])
        generated_value = 0.0
        if selection <= 0:
            generated_value = cls._dist_cos(w[1:], timestamp, minimum, maximum)
        if selection == 1:
            generated_value = cls._dist_linear(w[1:], timestamp, minimum, maximum)
        if selection >= 2:
            generated_value = cls._dist_quadratic(w[1:], timestamp, minimum, maximum)
        return float(generated_value)

    @classmethod
    def _dist_cos(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:  WPS123
        """
        Generate a value based on a cosine function.

        Parameters
        ----------
        w : NumberArray
            Behaviour defining vector of values between 0 and 1.
            w[0]: share of the interval used for shifting the cosine along the time axis.
            w[1]: defining periodical interval between 5 and 50.
            w[2]: share of the [feature_min, feature_max] interval, used for value axis shift.
            w[3]: proportional to cosine amplitudes.
        timestamp : float
            Point in time for which the function value of the transformed cosine is asked.
        minimum : float
            Minimal function value that is allowed to be generated.
        maximum : float
            Maximal function value that is allowed to be generated.

        Returns
        -------
        float
            The generated value.
        """
        interval = 45 * w[1] + 5
        shift = w[0] * interval
        average = w[2] * (maximum - minimum) + minimum
        amplitude = min(maximum - average, average - minimum) * w[3]

        timestamp = (timestamp + shift) * math.pi
        timestamp = timestamp % interval
        return float((np.cos(timestamp) * amplitude) + average)

    @classmethod
    def _dist_linear(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:  WPS123
        """
        Generate a value based on a linear function.

        Parameters
        ----------
        w : NumberArray
            Behaviour defining vector of values between 0 and 1.
            w[0]: share of the interval used for shifting the function along the time axis.
            w[1]: defining periodical interval between 5 and 50.
            w[2]: share of the [feature_min, feature_max] interval, used for value axis shift.
            w[3]: proportional to pitch of the linear function.
            w[4]: used to determine whether the pitch of the linear function is positive or negative.
        timestamp : float
            Point in time for which the function value of the transformed linear function is asked.
        minimum : float
            Minimal function value that is allowed to be generated.
        maximum : float
            Maximal function value that is allowed to be generated.

        Returns
        -------
        float
            The generated value.
        """
        sign = (math.floor(w[4] - 0.5) * 2) + 1
        interval = 45 * w[1] + 5
        shift = w[0] * interval
        base = w[2] * (maximum - minimum) + minimum
        amplitude = (maximum - base) / interval * w[3]

        timestamp = (timestamp + shift) % interval
        if sign == -1:
            base = base + (maximum - base)
        return float(timestamp * sign * amplitude + base)

    @classmethod
    def _dist_quadratic(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:WPS123
        """
        Generate a value based on a quadratic function.

        Parameters
        ----------
        w : NumberArray
            Behaviour defining vector of values between 0 and 1.
            w[0]: share of the interval used for shifting the function along the time axis.
            w[1]: defining periodical interval between 5 and 50.
            w[2]: share of the [feature_min, feature_max] interval, used for value axis shift.
            w[3]: proportional to pitch of the quadratic function.
            w[4]: used to determine whether the pitch of the quadratic function is positive or negative.
        timestamp : float
            Point in time for which the function value of the transformed quadratic function is asked.
        minimum : float
            Minimal function value that is allowed to be generated.
        maximum : float
            Maximal function value that is allowed to be generated.

        Returns
        -------
        float
            The generated value.
        """
        sign = (math.floor(w[4] - 0.5) * 2) + 1
        interval = 45 * w[1] + 5
        shift = w[0] * interval
        base = w[2] * (maximum - minimum) + minimum
        amplitude = (maximum - base) / (interval ** 2) * w[3]

        timestamp = (timestamp + shift) % interval
        if sign == -1:
            base = base + (maximum - base)
        return float((timestamp ** 2) * sign * amplitude + base)
