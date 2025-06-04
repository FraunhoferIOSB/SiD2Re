import random

from sid2re.driftgenerator.concept.nodes.root_distributions.base_distribution import BaseDistribution
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class GaussianDistribution(BaseDistribution):
    """Value generating distribution used in root features.

    This distribution generates random values based on a gaussian distribution.
    """

    @classmethod
    def _generate(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:  WPS123
        """
        Generate a random value using a Gaussian distribution.

        Parameters
        ----------
        w : NumberArray
            Behaviour defining vector of values between 0 and 1.
            w[0]: centre of the uniform distribution.
            w[1]: share of [minimum, maximum] that is used as variance for the Gaussian distribution.
        timestamp : float
            Point in time for which the function value of the function is asked (not used in this case).
        minimum : float
            Minimal function value that is allowed to be generated.
        maximum : float
            Maximal function value that is allowed to be generated.

        Returns
        -------
        float
            The random value sampled from the Gaussian distribution defined by w.
        """
        generated_value = random.gauss(w[0] * (maximum - minimum) + minimum, w[1] * (maximum - minimum) / 10)
        if generated_value > maximum:
            generated_value = maximum
        elif generated_value < minimum:
            generated_value = minimum
        return float(generated_value)
