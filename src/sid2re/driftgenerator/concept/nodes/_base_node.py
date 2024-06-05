from abc import ABC
from typing import List

import numpy as np

from sid2re.driftgenerator.concept.drift_behaviours import _BaseDriftBehaviour
from sid2re.driftgenerator.concept.drift_transition_functions import _BaseTransitionFunction, LinearTransitionFunction
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class _BaseNode(ABC):

    def __init__(self) -> None:
        """
        Initializes a node.
        """

        self.config: NumberArray = np.random.rand(10)
        self.trans: _BaseTransitionFunction = LinearTransitionFunction()
        self.shift_stamps: np.ndarray = np.array([])
        self.shift_info: np.ndarray = np.array([])
        self.drifts: List[_BaseDriftBehaviour] = []

    def set_transition(self, func: _BaseTransitionFunction) -> None:
        """
        Sets the transition function.

        Parameters
        ----------
        func : _BaseTransitionFunction
            The transition function.
        """

        self.trans = func

    def current_config(self, time: float) -> np.ndarray:
        """
        Calculates the current configuration.

        Parameters
        ----------
        time : float
            The current time.

        Returns
        -------
        np.ndarray
            The current configuration.
        """

        coeff = self.config
        for drift in self.drifts:
            coeff = drift.modify_coeff(coefficient=coeff, transition_behaviour=self.trans, current_time=time)
        return coeff

    def set_drift(self, drifts: List[_BaseDriftBehaviour]) -> None:
        """Sets the drift based on the provided shifts, shift_info, and index.

        Parameters
        ----------
        drifts : List[_BaseDriftBehaviour]
            List of drift behaviors.

        Raises
        ------
        TypeError
            If drifts is not a list.
        """
        if not isinstance(drifts, list):
            raise TypeError("Drifts must be a list.")
        self.drifts = drifts
