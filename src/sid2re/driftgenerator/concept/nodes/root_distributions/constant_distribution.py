from sid2re.driftgenerator.concept.nodes.root_distributions._base_distribution import _BaseDistribution

from sid2re.driftgenerator.utils.type_aliases import NumberArray


class ConstantDistribution(_BaseDistribution):
    """Value generating distribution used in root features. This distribution only generates one constant value."""

    @classmethod
    def _generate(cls, w: NumberArray, timestamp: float, minimum: float, maximum: float) -> float:  # noqa:  WPS123
        """
        Generate a constant value.

        Parameters
        ----------
        w : NumberArray
            Behaviour defining vector of values between 0 and 1.
            w[0]: value in [minimum, maximum] that used as constant function value.
        timestamp : float
            Point in time for which the function value of the function is asked (not used in this case).
        minimum : float
            Minimal function value that is allowed to be generated.
        maximum : float
            Maximal function value that is allowed to be generated.

        Returns
        -------
        float
            The constant value sampled from the uniform distribution defined by w.
        """
        generated_value = (w[0] * (maximum - minimum)) + minimum
        return float(generated_value)
