import random

from sid2re.driftgenerator.concept.nodes.root_distributions.base_distribution import BaseDistribution
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class UniformDistribution(BaseDistribution):
    """Value generating distribution used in root features.

    This distribution randomly generates values based on a uniform distribution.
    """

    @classmethod
    def _generate(  # noqa: WPS123,W0613,
        cls,
        w: NumberArray,  # noqa: WPS111
        timestamp: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """Generate a value based on a uniform distribution.

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
        generated_value = random.random() * w[1] * (maximum - minimum)
        max_val = w[1] * (maximum - minimum)
        centered_val = generated_value - max_val / 2
        offset = (w[0] * ((maximum - max_val / 2) - (minimum + max_val / 2))) + (minimum + max_val / 2)
        return float(centered_val + offset)
