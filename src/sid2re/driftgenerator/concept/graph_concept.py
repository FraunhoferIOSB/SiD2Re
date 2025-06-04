# pylint: disable=too-many-positional-arguments
import random
from typing import Any, Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sid2re.driftgenerator.concept.drift_behaviours import BaseDriftBehaviour
from sid2re.driftgenerator.concept.drift_transition_functions import BaseTransitionFunction
from sid2re.driftgenerator.concept.nodes import RandomConceptFunctionNode, RandomFunctionNode, RootFeatureNode
from sid2re.driftgenerator.concept.nodes.root_distributions import BaseDistribution
from sid2re.driftgenerator.utils.type_aliases import NumberArray


class ConceptGraph:
    """Build the networkx graph used to express the concept dependencies.

    This graph includes information of depth of dependencies, what are root nodes and what are targets.
    This is the base of truth for the data generation process.
    """

    def __init__(
        self,
        number_of_features: NumberArray,
        number_of_outputs: int,
        number_of_models: int,
        number_of_dependency_models: int,
        min_number_dependencies: int,
        max_number_dependencies: int,
        level_limited: bool,
        limit_target_dep: bool,
        n_target_dep: int,
        feature_min: float = -100.0,
        feature_max: float = 100.0,
    ):
        """Initialize a ConceptGraph instance.

        Parameters
        ----------
        number_of_features : NumberArray
            Array indicating the number of features at each level.

        number_of_outputs : int
            Number of output nodes.

        number_of_models : int
            Number of models.

        number_of_dependency_models : int
            Number of dependency models.

        min_number_dependencies : int
            Minimum number of dependencies for each node.

        max_number_dependencies : int
            Maximum number of dependencies for each node.

        level_limited : bool
            Whether the dependencies are limited by the level.

        limit_target_dep : bool
            Whether to limit the target dependencies.

        n_target_dep : int
            Number of target dependencies.

        feature_min : float
            Minimum value (default is -100.0).

        feature_max : float
            Maximum value (default is 100.0).
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self.number_of_features: np.ndarray = number_of_features
        self.node_pos: Dict[str, Tuple[int, int]] = {}
        self.r_map: Dict[str, RootFeatureNode] = {}
        self.fu_map: Dict[str, RandomFunctionNode] = {}
        self.c_map: Dict[str, RandomConceptFunctionNode] = {}
        self.f_map: Dict[str, Union[RootFeatureNode, RandomFunctionNode, RandomConceptFunctionNode]] = {}
        self.data_map: Dict[str, np.ndarray] = {}
        self.level_nodes: List[List[str]] = []
        self.feature_idx: List[str] = []
        self.target_nodes: List[str] = []
        current_node: int = 0

        # Add Nodes to the Graph
        for level_idx in range(number_of_features.shape[0]):
            level_nodes = []
            for feature_idx in range(number_of_features[level_idx]):
                self._graph.add_node(f'{level_idx}_{current_node}')
                self.node_pos[f'{level_idx}_{current_node}'] = (feature_idx, level_idx)
                if level_idx == 0:
                    self.r_map[f'{level_idx}_{current_node}'] = RootFeatureNode(feature_min, feature_max)
                    self.f_map[f'{level_idx}_{current_node}'] = self.r_map[f'{level_idx}_{current_node}']
                level_nodes += [f'{level_idx}_{current_node}']
                self.feature_idx += [f'{level_idx}_{current_node}']
                # Do later with more information
                current_node += 1
            self.level_nodes.append(level_nodes)
        self._graph.add_node('concept')
        self.node_pos['concept'] = (0, number_of_features.shape[0])
        self.c_map['concept'] = RandomConceptFunctionNode(number_of_models, n_target_dep, number_of_outputs)
        self.f_map['concept'] = self.c_map['concept']
        # Generate dependencies
        for node in list(self._graph.nodes()):
            # List possible dependencies
            if '0_' in node:
                continue
            if node == 'concept':
                num_dependencies = n_target_dep
                if limit_target_dep:
                    potential_depend = [
                        potential_parent_node for potential_parent_node in list(self._graph.nodes()) if
                        f'{number_of_features.shape[0] - 1}_' in potential_parent_node
                    ]
                else:
                    for feature_idx in range(number_of_features.shape[0]):
                        potential_depend += [
                            potential_parent_node for potential_parent_node in list(self._graph.nodes())
                            if f'{feature_idx}_' in potential_parent_node
                        ]
            else:
                num_dependencies = random.randint(min_number_dependencies, max_number_dependencies)
                if level_limited:
                    level = node.split('_')[0]
                    potential_depend = [
                        potential_parent_node for potential_parent_node in list(self._graph.nodes()) if
                        f'{int(level) - 1}_' in potential_parent_node
                    ]
                else:
                    level = node.split('_')[0]
                    potential_depend = []
                    for feature_idx in range(int(level)):
                        potential_depend += [
                            potential_parent_node for potential_parent_node in list(self._graph.nodes()) if
                            f'{feature_idx}_' in potential_parent_node
                        ]
            num_dependencies = min(num_dependencies, len(potential_depend))
            if node != 'concept' and '0_' not in node:
                self.fu_map[node] = RandomFunctionNode(number_of_dependency_models, num_dependencies)
                self.f_map[node] = self.fu_map[node]
            # Select and include dependencies into graph
            dependencies = random.sample(potential_depend, k=num_dependencies)
            for dep in dependencies:
                self._graph.add_edge(dep, node)
        # For visuals, also add Output level_nodes
        for feature_idx in range(number_of_outputs):
            self._graph.add_node(f'O_{feature_idx}')
            self.node_pos[f'O_{feature_idx}'] = (feature_idx, number_of_features.shape[0] + 1)
            self._graph.add_edge('concept', f'O_{feature_idx}')
            self.target_nodes += [f'O_{feature_idx}']

    def define_root_distros(self, declar: Dict[int, BaseDistribution]) -> None:
        """Define the root distributions for specified nodes.

        Parameters
        ----------
        declar : Dict[int, BaseDistribution]
            Dictionary where keys are node indices and values are distributions.
        """
        for idx in declar.keys():
            self.r_map[f'0_{idx}'].set_distro(declar[idx])

    def define_transition_func(self, func: BaseTransitionFunction) -> None:
        """Define the transition function for nodes in the graph.

        Parameters
        ----------
        func : BaseTransitionFunction
            Transition function.
        """
        for level in self.level_nodes:
            for node in level:
                self.f_map[node].set_transition(func)

    def generate_level_data(self, time_stamps: NumberArray, level: int) -> None:
        """Generate data for nodes at the specified level.

        Parameters
        ----------
        time_stamps : NumberArray
            Array of time stamps.

        level : int
            Level index.
        """
        self.data_map['time_idx'] = time_stamps
        if level > 0:
            for node in self.level_nodes[level]:
                self.data_map[node] = self.fu_map[node].generate_data(
                    time_stamps,
                    inputs=[
                        self.data_map[pred_node] for pred_node in self._graph.predecessors(node)
                    ],
                )
        else:
            for node in self.level_nodes[level]:
                self.data_map[node] = self.r_map[node].generate_data(time_stamps)

    def generate_output(self, time_stamps: NumberArray) -> None:
        """Generate output data for the concept.

        Parameters
        ----------
        time_stamps : NumberArray
            Array of time stamps.
        """
        predecessors_data = [self.data_map[pred_node] for pred_node in self._graph.predecessors('concept')]
        output = self.c_map['concept'].generate_data(time_stamps, inputs=predecessors_data)
        output = np.array(output)
        if len(output.shape) == 1:
            for node in self.target_nodes:
                self.data_map[node] = output
        else:
            for node_idx, node in enumerate(self.target_nodes):
                self.data_map[node] = output[:, node_idx]

    def pd_data_readout(self) -> pd.DataFrame:
        """Convert data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the data.
        """
        return pd.DataFrame.from_dict(self.data_map)

    def define_drift(
        self, concept_shift_info: List[BaseDriftBehaviour],
        data_shift_info: Dict[str, List[BaseDriftBehaviour]],
    ) -> None:
        """Define drift for nodes in the graph.

        Parameters
        ----------
        concept_shift_info : List[BaseDriftBehaviour]
            Concept shift information.

        data_shift_info : Dict[str, List[BaseDriftBehaviour]]
            Data shift information.
        """
        for level in self.level_nodes:
            for node in level:
                self.f_map[node].set_drift(data_shift_info[node])
        self.f_map['concept'].set_drift(concept_shift_info)

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying graph.

        Returns
        -------
        nx.DiGraph
            Directed graph representing the concept.
        """
        return self._graph

    def plot_graph(self) -> None:
        """Plot the graph for visualization."""
        plt.subplot(111)
        nx.draw(self._graph, pos=self.node_pos, with_labels=True, font_weight='bold')
        plt.show()


def _analyze_graph(graph: nx.DiGraph, output_nodes: List) -> Tuple[NumberArray, List]:
    roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    graph.remove_nodes_from(output_nodes)
    last_level = roots
    levels = [roots]
    while graph.nodes:
        graph.remove_nodes_from(last_level)
        current_level = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        levels += [current_level]
        last_level = current_level
    return np.array([len(level) for level in levels]), levels


class ConceptReader(ConceptGraph):
    """Concept graph representation that is inferred from a provided networkx graph.

    This will try to infer information from the provided structure, but might need additional specifications if the
    structure is not unambiguously.
    """

    def __init__(
        self, input_graph: nx.DiGraph, number_of_models: int, number_of_dependency_models: int,
        feature_min: float, feature_max: float, output_nodes: List[Any],
    ):  # pylint: disable=super-init-not-called
        """Initialize a ConceptReader instance.

        Parameters
        ----------
        input_graph : nx.DiGraph
            The input graph.
        number_of_models : int
            Number of models.
        number_of_dependency_models : int
            Number of dependency models.
        feature_min : float
            Minimum feature value.
        feature_max : float
            Maximum feature value.
        output_nodes : List[Any]
            List of output nodes.
        """
        self.r_map: Dict[str, RootFeatureNode] = {}
        self.fu_map: Dict[str, RandomFunctionNode] = {}
        self.c_map: Dict[str, RandomConceptFunctionNode] = {}
        self.f_map: Dict[str, Union[RootFeatureNode, RandomFunctionNode, RandomConceptFunctionNode]] = {}
        self._graph = nx.DiGraph()
        number_of_features, hierachy_list = _analyze_graph(input_graph.copy(), output_nodes)
        self.number_of_features = number_of_features
        self.node_pos = {}
        self.old_to_new_map = {}
        self.data_map = {}
        self.level_nodes = []
        self.feature_idx = []
        self.target_nodes = []
        current_node = 0
        # Add Nodes to the Graph
        for level_idx in range(number_of_features.shape[0]):
            level_nodes = []
            for feature_idx, node in zip(range(number_of_features[level_idx]), hierachy_list[level_idx]):
                self._graph.add_node(f'{level_idx}_{current_node}')
                self.old_to_new_map[node] = f'{level_idx}_{current_node}'
                self.node_pos[f'{level_idx}_{current_node}'] = (feature_idx, level_idx)
                if level_idx == 0:
                    self.r_map[f'{level_idx}_{current_node}'] = RootFeatureNode(feature_min, feature_max)
                    self.f_map[f'{level_idx}_{current_node}'] = self.r_map[f'{level_idx}_{current_node}']
                else:
                    self.fu_map[f'{level_idx}_{current_node}'] = (
                        RandomFunctionNode(number_of_dependency_models, num_dependencies=input_graph.in_degree(node))
                    )
                    self.f_map[f'{level_idx}_{current_node}'] = self.fu_map[f'{level_idx}_{current_node}']
                level_nodes += [f'{level_idx}_{current_node}']
                self.feature_idx += [f'{level_idx}_{current_node}']
                current_node += 1
            self.level_nodes.append(level_nodes)
        # Generate dependencies
        for level in hierachy_list:
            for node in level:
                for predec in input_graph.predecessors(node):
                    self._graph.add_edge(self.old_to_new_map[predec], self.old_to_new_map[node])
        concept_inputs = []
        for output in output_nodes:
            for predec in input_graph.predecessors(output):
                concept_inputs += [predec]
        concept_inputs = list(set(concept_inputs))
        self._graph.add_node('concept')
        self.node_pos['concept'] = (0, number_of_features.shape[0])
        self.c_map['concept'] = RandomConceptFunctionNode(number_of_models, len(concept_inputs), len(output_nodes))
        self.f_map['concept'] = self.c_map['concept']
        for input_values in concept_inputs:
            self._graph.add_edge(self.old_to_new_map[input_values], 'concept')
        # For visuals, also add Output level_nodes
        for feature_idx, _ in enumerate(output_nodes):
            self._graph.add_node(f'O_{feature_idx}')
            self.node_pos[f'O_{feature_idx}'] = (feature_idx, number_of_features.shape[0] + 1)
            self._graph.add_edge('concept', f'O_{feature_idx}')
            self.target_nodes += [f'O_{feature_idx}']
        print('The input graph was remapped to confine to the internal notations. The following maps old to new nodes')
        print(self.old_to_new_map)

    @property
    def number_of_features_per_level(self) -> NumberArray:
        """Fetch and count the number of features per dependency level.

        Returns
        -------
        NumberArray
            An array that points out the number of features for each dependency level. Starting with the root nodes
            at index 0.
        """
        return np.array(self.number_of_features)
