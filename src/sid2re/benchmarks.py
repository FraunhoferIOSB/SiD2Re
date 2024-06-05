from sid2re.driftgenerator.generator import DataGenerator
from sklearn.model_selection import ParameterGrid
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Seaborn visualization library
import seaborn as sns
import pandas as pd
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import warnings
import pkg_resources

warnings.filterwarnings("ignore", category=UserWarning)


def generate_for_pars(pars):
    generator = DataGenerator(number_of_models=pars['n_models'],
                              number_of_features=pars['n_features'],
                              number_of_outputs=pars['n_targets'],
                              concept_drifts=pars['n_drifts'][0],
                              data_drifts=pars['n_drifts'][1],
                              concept_drift_class=pars['concept_drift_class'],
                              data_drift_class=pars['data_drift_class'],
                              number_of_data_points=pars['n_data_points'],
                              noise_var=pars['noise'],
                              continous_time=pars['continous_time'],
                              rand_seed=pars['seed'],
                              drift_blocking_mode=pars['drift_blocking'],
                              max_severity=1,
                              min_severity=1)
    X, y = generator.get_data(n_uniform_feat=pars['feature_distributions'][0],
                              n_gauss_feat=pars['feature_distributions'][1],
                              n_constant_feat=pars['feature_distributions'][2],
                              n_periodical_feat=pars['feature_distributions'][3],
                              correlated_features=pars['feature_distributions'][4],
                              n_sensor_features=pars['feature_distributions'][5])
    concept_shift_information, data_shift_information = generator.get_shift_information()
    time_stamps = X['time_stamp']
    X = X.set_index(X['time_stamp']).drop(columns=['time_stamp'])
    y = y.set_index(y['time_stamp']).reset_index(drop=True).drop(columns=['time_stamp'])
    return concept_shift_information, data_shift_information, X, y, time_stamps


def save_to_disk(concept_shift_information, data_shift_information, X, y, time_stamps, pars, plot, pca_plot, location,
                 name):
    key = '_'.join([str(item) for item in pars.values()])
    name = f"concept_{pars['seed']}_{name}_{pars['n_features']}_{pars['n_targets']}"
    Path(f"{location}/datasets/{name}/{key}").mkdir(parents=True, exist_ok=True)
    with open(f'{location}/datasets/{name}/{key}/x.csv', "w") as f:
        X.to_csv(f, index=False)
    with open(f'{location}/datasets/{name}/{key}/y.csv', "w") as f:
        y.to_csv(f, index=False)
    with open(f'{location}/datasets/{name}/{key}/cd.csv', "w") as f:
        concept_shift_information.to_csv(f, index=False)
    with open(f'{location}/datasets/{name}/{key}/dd.csv', "w") as f:
        data_shift_information.to_csv(f, index=False)
    if plot:
        plot = sns.pairplot(data=pd.merge(X, y, left_index=True, right_index=True, how='left'),
                            plot_kws=dict(hue=time_stamps, palette="blend:darkblue,orange", edgecolor=None,
                                          size=0.1, alpha=0.75),
                            diag_kind='kde')
        plt.savefig(f"{location}/datasets/{name}/{key}/plot.png")
        plt.close()
    if pca_plot:
        pca = PCA(n_components=2, svd_solver='full')
        X = pca.fit_transform(X)
        pca = PCA(n_components=1, svd_solver='full')
        y = pca.fit_transform(y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, c=time_stamps, marker=".")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("target")
        plt.savefig(f"{location}/datasets/{name}/{key}/pca_3d_proj.png")
        plt.close()


def check_existence(pars, plot, pca_plot, name, location):
    key = '_'.join([str(item) for item in pars.values()])
    name = f"concept_{pars['seed']}_{name}_{pars['n_features']}_{pars['n_targets']}"

    if os.path.exists(Path(f"{location}datasets/{name}/{key}")) and len(
            os.listdir(Path(f"{location}datasets/{name}/{key}"))) >= (4 + plot + pca_plot):
        return True
    else:
        return False


def _async_loop(pars, plot, pca_plot, location, name, checkpointing):
    if checkpointing:
        if check_existence(pars, plot, pca_plot, name, location):
            return
    concept_shift_information, data_shift_information, X, y, time_stamps = generate_for_pars(pars)
    save_to_disk(concept_shift_information, data_shift_information, X, y, time_stamps, pars, plot, pca_plot,
                 location, name)
    return


def _async_wrap(param_grid, plot, pca_plot, location, name, checkpointing, max_workers):
    number_of_tasks = len(list(ParameterGrid(param_grid)))
    plot = [plot] * number_of_tasks
    pca_plot = [pca_plot] * number_of_tasks
    location = [location] * number_of_tasks
    name = [name] * number_of_tasks
    checkpointing = [checkpointing] * number_of_tasks
    _ = process_map(_async_loop,
                    *[list(ParameterGrid(param_grid)), plot, pca_plot, location, name, checkpointing],
                    max_workers=max_workers)


def _execute_generation(param_grid, plot, pca_plot, location, name, checkpointing, multi_thread, max_workers):
    if multi_thread:
        if plot or pca_plot:
            raise RuntimeError(
                "Execution using parallel computations is not supported with simultaneous plotting")
        _async_wrap(param_grid, plot, pca_plot, location, name, checkpointing, max_workers)
    else:
        for pars in tqdm(list(ParameterGrid(param_grid)),
                         bar_format=f"Generating concept_{param_grid['seed'][0]}_{name}_"
                                    f"{param_grid['n_features'][0]}_{param_grid['n_targets'][0]}"
                                    "||Eta:{eta}.|{bar}{r_bar}"):
            if checkpointing:
                if check_existence(pars, plot, pca_plot, name, location):
                    continue
            concept_shift_information, data_shift_information, X, y, time_stamps = generate_for_pars(pars)
            save_to_disk(concept_shift_information, data_shift_information, X, y, time_stamps, pars, plot,
                         pca_plot,
                         location,
                         name)


def generate_benchmark_v1(location="./", plot=True, pca_plot=True, checkpointing=True, generate_small=True,
                          generate_standard=True, generate_dependent=True, multi_thread=False, max_workers=2):
    if pkg_resources.get_distribution("sid2regenerator").version != "0.1.1":
        raise RuntimeError(
            "This benchmark is based on the sid2re version 0.1.1. "
            f"To ensure reproducibility generating older benchmarks with the currently installed version "
            f"{pkg_resources.get_distribution('sid2regenerator').version} is not allowed.")
        return
    if generate_small:
        seeds = [2, 3, 4, 9, 10, 13, 20, 26]
        names = ['wave', 'fold', 'skew_raise', 'broken_helix', 'steps', 'sideways_helix', 'double_fold', 'layered_peak']
        for current_seed, name in zip(seeds, names):
            param_grid = {'n_data_points': [500, 1000, 10000],
                          'n_drifts': [[0, 0], [1, 0], [2, 0], [1, 1]],
                          'concept_drift_class': ['sudden', 'gradual'],
                          'data_drift_class': ['incremental', 'faulty_sensor'],
                          'drift_blocking': [True, False],
                          'continous_time': [True],
                          'noise': [0, 0.001, 0.1],
                          'feature_distributions': [[2, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 2, 0, 0]],
                          'n_features': [2],
                          'n_targets': [1],
                          'n_models': [5],
                          'seed': [current_seed]}
            _execute_generation(param_grid, plot, pca_plot, location, name, checkpointing, multi_thread, max_workers)
    if generate_standard:
        seeds = [10, 15, 13, 6, 9]
        names = ['multi_dim_tree', 'multi_dim_fold', 'multi_dim_splitter_tree', 'polynomial', 'multi_dim_hook']
        for current_seed, name in zip(seeds, names):
            param_grid = {'n_data_points': [500, 1000, 10000],
                          'n_drifts': [[0, 0], [1, 0], [2, 0], [1, 1]],
                          'concept_drift_class': ['sudden', 'gradual'],
                          'data_drift_class': ['incremental', 'faulty_sensor'],
                          'drift_blocking': [True, False],
                          'continous_time': [True],
                          'noise': [0, 0.001, 0.1],
                          'feature_distributions': [[0, 4, 0, 0, 0, 0],
                                                    [2, 0, 0, 2, 0, 0],
                                                    [2, 2, 0, 0, 2, 0],
                                                    [4, 0, 0, 0, 0, 0]],
                          'n_features': [4],
                          'n_targets': [2],
                          'n_models': [3],
                          'seed': [current_seed]}
            _execute_generation(param_grid, plot, pca_plot, location, name, checkpointing, multi_thread, max_workers)
    if generate_dependent:
        seeds = [5, 1, 2, 8, 19, 24]
        names = ['helix_abstraction_dependent', 'relu_cut_off_dependent', 'splintered_spline_dependent',
                 'clustered_polynomial_dependent', 'unbalanced_tree_dependent', 'splintered_step_dependent']
        for current_seed, name in zip(seeds, names):
            param_grid = {'n_data_points': [500, 1000, 10000],
                          'n_drifts': [[0, 0], [1, 0], [2, 0], [1, 1]],
                          'concept_drift_class': ['sudden', 'gradual'],
                          'data_drift_class': ['incremental', 'faulty_sensor'],
                          'drift_blocking': [True, False],
                          'continous_time': [True],
                          'noise': [0, 0.001, 0.1],
                          'feature_distributions': [[2, 0, 0, 0, 0, 2],
                                                    [0, 2, 0, 0, 0, 2],
                                                    [0, 0, 2, 0, 0, 2],
                                                    [0, 0, 0, 2, 0, 2]],
                          'n_features': [4],
                          'n_targets': [2],
                          'n_models': [3],
                          'seed': [current_seed]}
            _execute_generation(param_grid, plot, pca_plot, location, name, checkpointing, multi_thread, max_workers)
