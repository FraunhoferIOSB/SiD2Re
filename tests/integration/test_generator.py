from sid2re.driftgenerator.generator import DataGeneratorGraph
from sid2re.driftgenerator.concept.drift_behaviours import SuddenDriftBehaviour, IncrementalDriftBehaviour, \
    GradualDriftBehaviour, FaultySensorDriftBehaviour,ReoccuringSuddenDriftBehaviour,ReoccuringIncrementalDriftBehaviour,ReoccuringFaultySensorDriftBehaviour,ReoccuringGradualDriftBehaviour
import numpy as np
import pytest
import networkx as nx


def test_standard_run():
    generator = DataGeneratorGraph(number_of_models=5,
                                   number_of_dependency_models=2,
                                   n_target_dep=2,
                                   number_of_features=np.array([4, 3]),
                                   number_of_outputs=2,
                                   concept_drifts=1,
                                   data_drifts=1,
                                   number_of_data_points=500,
                                   noise_var=0.1,
                                   continuous_time=True,
                                   rand_seed=1,
                                   drift_blocking_mode=True,
                                   max_severity=1,
                                   min_severity=1,
                                   root_distros=[1, 1, 1, 1])
    data_df = generator.get_data()
    print(data_df)
    concept_shift_information, data_shift_information = generator.get_shift_information()
    concept = generator.concept
    _ = generator.get_concept_adjacency_matrix()
    _ = generator.get_concept_adjacency_matrix(output_node_names=True)
    print(concept_shift_information)
    print(data_shift_information)
    assert data_df.size == 500 * 10


@pytest.mark.parametrize("n_cd_drifts", [0, 1, 2])
@pytest.mark.parametrize("n_dd_drifts", [0, 1, 2])
@pytest.mark.parametrize("cd_class", [SuddenDriftBehaviour, IncrementalDriftBehaviour])
@pytest.mark.parametrize("dd_class", [GradualDriftBehaviour, FaultySensorDriftBehaviour])
@pytest.mark.parametrize("bool_switch", [False, True])
def test_parameter_grid(n_cd_drifts, n_dd_drifts, cd_class, dd_class, bool_switch):
    generator = DataGeneratorGraph(number_of_models=5,
                                   number_of_dependency_models=2,
                                   n_target_dep=2,
                                   number_of_features=np.array([4, 3]),
                                   number_of_outputs=2,
                                   concept_drifts=n_cd_drifts,
                                   data_drifts=n_dd_drifts,
                                   concept_drift_class=cd_class,
                                   data_drift_class=dd_class,
                                   number_of_data_points=500,
                                   noise_var=0.1,
                                   continuous_time=bool_switch,
                                   rand_seed=1,
                                   drift_blocking_mode=bool_switch,
                                   max_severity=1,
                                   min_severity=1,
                                   root_distros=[0, 1, 1, 1],
                                   level_limited=bool_switch,
                                   limit_target_dep=bool_switch,
                                   min_number_dependencies=1,
                                   max_number_dependencies=10
                                   )
    data_df = generator.get_data()
    concept_shift_information, data_shift_information = generator.get_shift_information()
    assert data_df.size == 500 * 10


def test_false_initialization():
    with pytest.raises(ValueError) as excinfo:
        _ = DataGeneratorGraph(number_of_models=5,
                               number_of_dependency_models=2,
                               n_target_dep=2,
                               number_of_features=np.array([4, 3]),
                               number_of_outputs=2,
                               concept_drifts=1,
                               data_drifts=1,
                               concept_drift_class=SuddenDriftBehaviour,
                               data_drift_class=IncrementalDriftBehaviour,
                               number_of_data_points=1000,
                               noise_var=0.1,
                               continuous_time=True,
                               rand_seed=1,
                               drift_blocking_mode=True,
                               max_severity=1,
                               min_severity=1,
                               root_distros=[4, 1, 1, 1])
    assert len(str(excinfo.value)) > 0


def test_graph_initialization():
    graph = nx.DiGraph()
    graph.add_node("feat_1")
    graph.add_node("feat_2")
    graph.add_node("feat_3")
    graph.add_node("feat_4")
    graph.add_edge("feat_1", "feat_3")
    graph.add_edge("feat_2", "feat_3")
    graph.add_node("label_1")
    graph.add_edge("feat_1", "label_1")
    graph.add_edge("feat_3", "label_1")
    graph.add_edge("feat_4", "label_1")
    generator = DataGeneratorGraph(graph=graph,
                                   output_nodes=["label_1"],
                                   number_of_models=5,
                                   number_of_dependency_models=2,
                                   number_of_outputs=2,
                                   concept_drifts=1,
                                   data_drifts=1,
                                   number_of_data_points=1000,
                                   noise_var=0.1,
                                   continuous_time=True,
                                   rand_seed=1,
                                   drift_blocking_mode=True,
                                   max_severity=1,
                                   min_severity=1,
                                   root_distros=[1, 1, 0, 0])
    data_df = generator.get_data()
    print(data_df)
    concept_shift_information, data_shift_information = generator.get_shift_information()
    assert data_df.size == 1000 * 6
