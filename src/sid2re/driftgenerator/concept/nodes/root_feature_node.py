"""Functionalities surrounding the root nodes that generate data inputs based on basic probability density functions."""
import numpy as np

from sid2re.driftgenerator.concept.nodes.base_node import BaseNode
from sid2re.driftgenerator.concept.nodes.root_distributions import BaseDistribution, UniformDistribution


class RootFeatureNode(BaseNode):
    """Noe representing an independent feature. Also called root feature."""

    def __init__(self, minimum: float, maximum: float):
        """
        Initialize a RootFeatureNode instance.

        Parameters
        ----------
        minimum : float
            The minimum value of the generated feature.
        maximum : float
            The maximum value of the generated feature.
        """
        super().__init__()
        self._minimum: float = minimum
        self._maximum: float = maximum
        self.distribution: BaseDistribution = UniformDistribution()

    def set_distro(self, distribution: BaseDistribution) -> None:
        """
        Set the distribution type.

        Parameters
        ----------
        distribution : BaseDistribution
            The distribution type.
        """
        self.distribution = distribution

    def generate_data(self, time_stamps: np.ndarray) -> np.ndarray:
        """
        Generate data based on the configured distribution.

        Parameters
        ----------
        time_stamps : np.ndarray
            List of time stamps.

        Returns
        -------
        np.ndarray
            List of generated data.
        """
        generated_data = [
            self.distribution.generate(self.current_config(time), time, self._minimum, self._maximum)
            for time in time_stamps
        ]
        return np.array(generated_data)
