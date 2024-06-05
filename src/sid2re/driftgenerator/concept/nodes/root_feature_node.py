import numpy as np

from sid2re.driftgenerator.concept.nodes._base_node import _BaseNode
from sid2re.driftgenerator.concept.nodes.root_distributions import _BaseDistribution, UniformDistribution


class RootFeatureNode(_BaseNode):
    """Noe representing an independent feature. Also called root feature."""

    def __init__(self, minimum: float, maximum: float):
        """
        Initializes a RootFeatureNode instance.

        Parameters
        ----------
        minimum : float
            The minimum value of the generated feature.
        maximum : float
            The maximum value of the generated feature.
        """
        super().__init__()
        self.minimum: float = minimum
        self.maximum: float = maximum
        self.distribution: _BaseDistribution = UniformDistribution()

    def set_distro(self, distribution: _BaseDistribution) -> None:
        """
        Sets the distribution type.

        Parameters
        ----------
        distribution : _BaseDistribution
            The distribution type.
        """
        self.distribution = distribution

    def generate_data(self, time_stamps: np.ndarray) -> np.ndarray:
        """
        Generates data based on the configured distribution.

        Parameters
        ----------
        time_stamps : np.ndarray
            List of time stamps.

        Returns
        -------
        np.ndarray
            List of generated data.
        """

        return np.array(
            [self.distribution.generate(self.current_config(time), time, self.minimum, self.maximum) for time in
             time_stamps]
        )
