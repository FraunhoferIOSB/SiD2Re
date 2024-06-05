"""Type aliases and definitions."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy import typing as npt

# Base Types
NumberArray = npt.NDArray[np.number]
DataTuple = Tuple[NumberArray, NumberArray]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]
Estimator = Any

# Type Compounds.
EnsDict = Dict[int, Estimator]
MetrOut = Union[float, npt.NDArray[np.float64]]
Metric = Callable[[NumberArray, NumberArray, Optional[bool]], MetrOut]
