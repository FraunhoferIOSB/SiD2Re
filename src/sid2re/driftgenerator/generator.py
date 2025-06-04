# pylint: disable=too-many-positional-arguments
"""Entrypoint for the fundamental generator capabilities.

Provides the basic construction procedure for node dependencies and handles data generation.
"""
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from sid2re.driftgenerator.concept.drift_behaviours import (
    BaseDriftBehaviour, FaultySensorDriftBehaviour,
    GradualDriftBehaviour, IncrementalDriftBehaviour, SuddenDriftBehaviour,
)
from sid2re.driftgenerator.concept.drift_transition_functions import BaseTransitionFunction, LinearTransitionFunction
from sid2re.driftgenerator.concept.graph_concept import ConceptGraph, ConceptReader
# Import internal functionalities
from sid2re.driftgenerator.concept.nodes.root_distributions import (
    BaseDistribution, ConstantDistribution,
    GaussianDistribution, PeriodicalDistribution, UniformDistribution,
)
from sid2re.driftgenerator.utils.type_aliases import NumberArray


def _distro_declarator(
    n_features: int,
    n_uniform_feat: int,
    n_gauss_feat: int,
    n_constant_feat: int,
    n_periodical_feat: int,
) -> Dict[int, BaseDistribution]:
    if n_uniform_feat + n_gauss_feat + n_constant_feat + n_periodical_feat > n_features:
        n_set_feat = n_uniform_feat + n_gauss_feat + n_constant_feat + n_periodical_feat
        raise ValueError(
            f'Too many root features set staticaly: {n_set_feat} of {n_features}',
        )
    declarator: Dict[int, BaseDistribution] = {}
    indices = np.arange(0, n_features)
    np.random.shuffle(indices)
    list_of_indices = list(indices)

    # set feature behaviour
    for _ in range(n_uniform_feat):
        declarator[list_of_indices.pop()] = UniformDistribution()
    for _ in range(n_gauss_feat):
        declarator[list_of_indices.pop()] = GaussianDistribution()
    for _ in range(n_constant_feat):
        declarator[list_of_indices.pop()] = ConstantDistribution()
    for _ in range(n_periodical_feat):
        declarator[list_of_indices.pop()] = PeriodicalDistribution()

    for _ in list_of_indices:
        declarator[list_of_indices.pop()] = random.choice(
            (UniformDistribution(), GaussianDistribution(), ConstantDistribution(), PeriodicalDistribution()),
        )

    return declarator


def _set_concept_drifts(
    rand_seed: float,
    drift_blocking_mode: bool,
    concept_drifts: int,
    time_stamp: NumberArray,
    min_severity: float,
    max_severity: float,
    number_of_models: int,
    concept_drift_class: Optional[BaseDriftBehaviour],
) -> List:
    drifts = []
    if drift_blocking_mode and concept_drifts != 0:
        step = (time_stamp[-1] - time_stamp[0]) / concept_drifts
        concept_shift_stamps = np.array([0.5 * step + step * cd_idx for cd_idx in range(concept_drifts)])
    else:
        concept_shift_stamps = np.random.rand(concept_drifts) * (time_stamp[-1] - time_stamp[0])
    if rand_seed != -1:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
    concept_shift_info = []
    if concept_shift_stamps.size > 0:
        for _ in range(concept_shift_stamps.shape[0]):
            if drift_blocking_mode:
                drift_spacing = (time_stamp[-1] - time_stamp[0]) / concept_drifts
                concept_shift_info.append(
                    [
                        random.random() / 2 * drift_spacing,
                        (
                            (np.minimum(
                                max_severity,
                                np.maximum(min_severity, np.random.rand(number_of_models)),
                            ) - 0.5) * 1000 / number_of_models  # noqa: WPS319
                        ),
                        random.choice(  # noqa: S311
                            [
                                FaultySensorDriftBehaviour, GradualDriftBehaviour,
                                IncrementalDriftBehaviour, SuddenDriftBehaviour,
                            ],
                        ),
                        random.choice([True, False]),
                    ],
                )
            else:
                concept_shift_info.append(
                    [
                        random.random() * (np.max(concept_shift_stamps) / 2),
                        (np.random.randint(1) * 2 - 1) * np.minimum(
                            max_severity, np.maximum(min_severity, np.random.rand(number_of_models)),
                        ) * 1000 / number_of_models,
                        random.choice(
                            [
                                FaultySensorDriftBehaviour, GradualDriftBehaviour,
                                IncrementalDriftBehaviour, SuddenDriftBehaviour,
                            ],
                        ),
                        random.choice([True, False]),
                    ],
                )
    for stamp, info in zip(concept_shift_stamps, concept_shift_info):
        if concept_drift_class is not None:
            cd_class = concept_drift_class
        else:
            cd_class = info[2]
        drifts += [cd_class(
            drift_time=stamp,
            drift_radius=info[0],
            coefficient_shift=info[1],
            reoccurring=info[3],
        )]  # type: ignore[operator]
    return drifts


def _set_data_drifts(
    rand_seed: int,
    drift_blocking_mode: bool,
    data_drifts: int,
    time_stamp: NumberArray,
    feature_idxs: List[str],
    min_severity: float,
    max_severity: float,
    data_drift_class: Optional[BaseDriftBehaviour],
) -> Dict[str, List[BaseDriftBehaviour]]:
    drifts: Dict[str, List[BaseDriftBehaviour]] = {}
    for idx in feature_idxs:
        drifts[idx] = []
    if drift_blocking_mode and data_drifts != 0:
        step = (time_stamp[-1] - time_stamp[0]) / data_drifts
        data_shifts = np.array([0.5 * step + step * dd_idx for dd_idx in range(data_drifts)])
    else:
        data_shifts = np.random.rand(data_drifts) * (time_stamp[-1] - time_stamp[0])
    if rand_seed != -1:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
    data_shift_info = []
    for _ in range(data_drifts):

        if drift_blocking_mode:
            data_shift_info.append(
                [
                    random.choice(feature_idxs),
                    random.random() / 2 * (time_stamp[-1] - time_stamp[0]) / data_drifts,
                    np.minimum(max_severity, np.maximum(min_severity, np.random.rand(10))),
                    random.choice(
                        [
                            FaultySensorDriftBehaviour, GradualDriftBehaviour,
                            IncrementalDriftBehaviour, SuddenDriftBehaviour,
                        ],
                    ),
                    random.choice([True, False]),
                ],
            )
        else:
            data_shift_info.append(
                [
                    random.choice(feature_idxs),
                    random.random() * (np.max(data_shifts) / 2),
                    np.minimum(max_severity, np.maximum(min_severity, np.random.rand(10))),
                    random.choice(
                        [
                            FaultySensorDriftBehaviour, GradualDriftBehaviour,
                            IncrementalDriftBehaviour, SuddenDriftBehaviour,
                        ],
                    ),
                    random.choice([True, False]),
                ],
            )
    for stamp, info in zip(data_shifts, data_shift_info):
        if data_drift_class is not None:
            dd_class = data_drift_class
        else:
            dd_class = info[3]
        drifts[info[0]] += [
            dd_class(
                drift_time=stamp,
                drift_radius=info[1],
                coefficient_shift=info[2],
                reoccurring=info[4],
            )]  # type: ignore[operator]
    return drifts


class DataGeneratorGraph:
    """
    Generator used to create a non-stationary regression problem.

    Initialization defines the behavior of the non-stationary concept, and sampling defines the behavior of the
    input data.
    """

    def __init__(
        self,
        number_of_data_points: int = 100,
        number_of_features: Optional[NumberArray] = None,
        number_of_outputs: int = 1,
        feature_min: float = -100.0,
        feature_max: float = 100.0,
        number_of_models: int = 5,
        noise_var: float = 0,
        rand_seed: int = -1,
        concept_drifts: int = 0,
        data_drifts: int = 0,
        concept_drift_class: Optional[BaseDriftBehaviour] = None,
        data_drift_class: Optional[BaseDriftBehaviour] = None,
        transition_func: Optional[BaseTransitionFunction] = None,
        continuous_time: bool = True,
        max_time_sparsity: int = 10,
        drift_blocking_mode: bool = False,
        max_severity: float = 1,
        min_severity: float = 0,
        number_of_dependency_models: int = 2,
        min_number_dependencies: int = 2,
        max_number_dependencies: int = 2,
        level_limited: bool = True,
        limit_target_dep: bool = True,
        n_target_dep: int = 1,
        root_distros: Optional[List[int]] = None,
        graph: Optional[nx.DiGraph] = None,
        output_nodes: Optional[List[Any]] = None,
    ) -> None:
        """
        Initialize the DataGenerator, defining the behavior of the non-stationary concept.

        Parameters
        ----------
        number_of_data_points : int
            Number of instances included in the dataset generated by `get_data()`.

        number_of_features : Optional[NumberArray]
            Number of input features generated and used in the data stream. Positions in the array define the different
            levels of dependency. The default value only generates two root features that do not have any dependencies.

        number_of_outputs : int
            Number of output targets produced by the artificial non-stationary concept, based on the generated input
            features.

        feature_min : float
            The initial set of generated input features will not fall below this value. Note that data drifts and noise
            may lead to feature values that do not comply with this limitation.

        feature_max : float
            The initial set of generated input features will not surpass this value. Note that data drifts and noise
            may lead to feature values that do not comply with this limitation.

        number_of_models : int
            Number of random models used to build the artificial concept. More models enable more complex concepts.
            Note that an increasing number of models does increase the maximal complexity but also may reduce variance
            in the generated concept, as the random models are linearly combined to form the concept.

        noise_var : float
            Variance of the normal distribution used to sample the scaling factor for noise in features and targets.
            As input and output values may have very different value ranges, we opted for a noisy scaling factor instead
            of an additive noise.

        rand_seed : int
            Seed used for random choices like sampling. If set to -1, no specific seed will be set, leading to random
            results in every execution.

        concept_drifts : int
            Number of concept drifts introduced in the generated dataset of `get_data()`.

        data_drifts : int
            Number of data drifts introduced in the generated dataset of `get_data()`.

        concept_drift_class : Optional[BaseDriftBehaviour]
            The choosen drift behaviour class (like 'sudden', 'gradual', 'incremental', ...) of the generated
            concept drifts.
            If set to None, the type will be decided for each drift at random.

        data_drift_class : Optional[BaseDriftBehaviour]
            The choosen drift behaviour class ('sudden', 'gradual', 'incremental', 'faulty_sensor', ...) of the
            generated data drifts.
            If set to None, the type will be decided for each drift at random.

        transition_func : Optional[BaseTransitionFunction]
            How two concepts are to be blended. Used for incremental and gradual features. Custom transition functions
            may be provided but must implement the API of the following exemplary call:
            `transition_func(current_time_stamp: int, shift_centre: int, shift_radius: float) -> float`.
            Note that the transition function is only called if there is a transition ongoing, i.e.:
            `shift_centre - shift_radius < current_time_stamp <= shift_centre + shift_radius`.

            See the exemplary `LinearTransitionFunction` used as default:
            ```
            def linear_trans_func(current_time, shift_centre, shift_radius):
                return (current_time - (shift_centre - shift_radius)) / (shift_radius * 2)
            ```

        continuous_time : bool
            Whether to use complete time series or time series with missing timestamps.
            If set to False, behavior of missing timestamps can be set by `max_time_sparsity`.

        max_time_sparsity : int
            Only used when `continuous_time=False`.
            Sets the maximal number of timestamps skipped after a timestamp is evaluated.

        drift_blocking_mode : bool
            Whether drifts may overlap or be confined to equally sized timeframes in the generated time series.
            If set to True, concept drifts will not overlap with other concept drifts; same goes for data drifts.
            Note that concept and data drifts are still able to overlap and influence each other.

        max_severity : float
            Internal scaling factor on how drastic drifts are allowed to be.
            Can be used to control the relative intensity of drifts.
            If `max_severity` = `min_severity`, then all drifts will have comparable impacts on data distributions and
            concept.

        min_severity : float
            Internal scaling factor on how drastic drifts have to be.

        number_of_dependency_models : int
            Number of dependency models, used between features of different dependency layers.

        min_number_dependencies : int
            Minimum number of feature dependencies.

        max_number_dependencies : int
            Maximum number of feature dependencies.

        level_limited : bool
            Whether the dependencies are limited to level l to level l-1 or not.

        limit_target_dep : bool
            Whether to limit target dependencies to the last feature layer.

        n_target_dep : int
            Number of target dependencies.

        root_distros : Optional[List[int]]
            List of root distributions. `[n_uniform_feat, n_gauss_feat, n_constant_feat, n_periodical_feat]`
            The input arguments can be used to set the minimal number of features that are sampled from a particular
            kind of distribution. You may only assign as many features as were defined in the first feature layer.
            If there are fewer given assignments than features, the remaining features will be sorted into categories
            by random.

        graph : Optional[nx.DiGraph]
            Concept graph.

        output_nodes : Optional[List[Any]]
            List of output nodes of the input graph.
        """
        if number_of_features is None:
            number_of_features = np.array([2])
        if root_distros is None:
            root_distros = [2, 0, 0, 0]
        if transition_func is None:
            transition_func = LinearTransitionFunction()
        if output_nodes is None:
            output_nodes = []
        # define time index
        if rand_seed != -1:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
        if continuous_time:
            time_stamp = np.arange(0, number_of_data_points)
        else:
            temp_time_stamp = []
            current_time_stamp = 0
            for _ in range(number_of_data_points):
                temp_time_stamp.append(current_time_stamp)
                current_time_stamp += random.randint(1, max_time_sparsity)
            time_stamp = np.array(temp_time_stamp)
        self._continous_time = continuous_time
        self._number_of_data_points = number_of_data_points
        self._time_stamp = time_stamp
        self._transition_func = transition_func
        # define concept graph
        if graph is None:
            self._concept = ConceptGraph(
                number_of_features, number_of_outputs, number_of_models,
                number_of_dependency_models, min_number_dependencies, max_number_dependencies,
                level_limited, limit_target_dep, n_target_dep, feature_min, feature_max,
            )
        else:
            self._concept = ConceptReader(
                graph, number_of_models, number_of_dependency_models, feature_min, feature_max,
                output_nodes,
            )
            number_of_features = self._concept.number_of_features_per_level
        declar = _distro_declarator(number_of_features[0], *root_distros)
        self._concept.define_root_distros(declar)
        self._concept.define_transition_func(transition_func)
        # define drifts
        self._concept_drifts = _set_concept_drifts(
            rand_seed, drift_blocking_mode,
            concept_drifts, time_stamp,
            min_severity, max_severity,
            number_of_models, concept_drift_class,
        )
        self._data_drifts = _set_data_drifts(
            rand_seed, drift_blocking_mode, data_drifts,
            time_stamp, self._concept.feature_idx,
            min_severity, max_severity, data_drift_class,
        )
        self._concept.define_drift(self._concept_drifts, self._data_drifts)
        self._feature_min = feature_min
        self._feature_max = feature_max
        self._n_models = number_of_models
        self._n_features = number_of_features
        self._n_outputs = number_of_outputs
        self._noise_var = noise_var

    def get_data(self) -> pd.DataFrame:
        """
        Use the initialized Generator to sample data points in the non-stationary environment.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the features and targets.

        Notes
        -----
        This method generates a dataset based on the non-stationary environment settings configured during the
        initialization of the `DataGenerator` instance. The data points are sampled according to the defined
        features, models, drifts, and other parameters, ensuring that the resulting dataset reflects the
        specified non-stationary behaviors.
        """
        for feature_idx in range(self._n_features.shape[0]):
            self._concept.generate_level_data(self._time_stamp, feature_idx)
        self._concept.generate_output(self._time_stamp)
        data_df = self._concept.pd_data_readout()
        data_df = data_df.multiply(
            np.concatenate(
                [
                    np.ones((data_df.shape[0], 1)),
                    np.random.normal(1, self._noise_var, (data_df.shape[0], data_df.shape[1] - 1)),
                ],
                axis=1,
            ),
        )
        return data_df  # noqa: WPS331

    @property
    def concept(self) -> ConceptGraph:
        """
        Provide the drift-induced concept function that is used to map the generated inputs to the targets.

        Returns
        -------
        ConceptGraph
            Concept used in the drifted datastream
        """
        return self._concept

    def get_concept_adjacency_matrix(
        self, output_node_names: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Any]]]:
        """
        Return the adjacency matrix of the concept graph.

        Parameters
        ----------
        output_node_names : bool
            Whether to include output node names.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, List[Any]]]
            If `output_node_names` is True, returns a tuple containing:
            - np.ndarray: The adjacency matrix of the concept graph.
            - List[Any]: A list of output node names.

            Otherwise, returns only the adjacency matrix (np.ndarray).

        Notes
        -----
        The adjacency matrix represents the structure of the concept graph where each element (i, j) indicates the
        presence of a directed edge from node i to node j. The `output_node_names` parameter can be used to retrieve
        the names of the nodes in the graph, which can be helpful for interpreting the structure of the graph.
        """
        if output_node_names:
            return np.asarray(
                nx.adjacency_matrix(self._concept.graph).todense(),
            ), self._concept.graph.nodes()
        return np.asarray(nx.adjacency_matrix(self._concept.graph).todense())

    def get_shift_information(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Provide two pandas dataframes, that describe the concept and data drifts.

        For the concept drifts (time, duration, weight shift, class) is provided.
        For data drifts (time, affected feature, duration, weight shift, class) is provided.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Drift Information DataFrames
        """
        concept_info = []
        for concept_drift in self._concept_drifts:
            concept_info += [concept_drift.drift_information]
        concept = pd.DataFrame(concept_info, columns=['time_stamp(centre)', 'radius', 'shift', 'class'])

        data_info = []
        for node_idx, entry in self._data_drifts.items():
            for data_drift in entry:
                data_info += [[*(data_drift.drift_information), node_idx]]
        features = pd.DataFrame(
            data_info,
            columns=[
                'time_stamp(centre)', 'radius', 'shift of distribution parameters', 'class', 'affected_feature',
            ],
        )
        return concept, features
