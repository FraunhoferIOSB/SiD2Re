# pylint: disable=unused-argument,too-many-positional-arguments
import hashlib
import itertools
import shutil
import warnings
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sid2re.driftgenerator.generator import DataGeneratorGraph
from sid2re.driftgenerator.utils.files import get_script_location

warnings.filterwarnings('ignore', category=UserWarning)


def _load_param_grid(grid_path: Path) -> List[Dict]:
    with open(grid_path, 'r', encoding='utf-8') as param_file:
        raw = yaml.safe_load(param_file)
    sweep_params = {key: param_value for key, param_value in raw.items() if key != 'fixed'}
    fixed_params = raw.get('fixed', {})
    dict_keys, dict_values = zip(*sweep_params.items())
    return [dict(zip(dict_keys, product_value), **fixed_params) for product_value in itertools.product(*dict_values)]


def _config_hash(config: dict) -> str:
    conf_str = yaml.dump(config, sort_keys=True)
    return hashlib.md5(conf_str.encode(), usedforsecurity=False).hexdigest()[:8]


def generate_benchmark_v1(
    location: str = './',
    plot: bool = True,
    pca_plot: bool = True,
    checkpointing: bool = True,
    generate_small: bool = True,
    generate_standard: bool = True,
    generate_dependent: bool = True,
    multi_thread: bool = False,
    max_workers: int = 2,
) -> None:
    """
    Generate benchmark datasets with different configurations.

    Parameters
    ----------
    location : str
        The directory where the generated datasets will be stored.
    plot : bool
        Whether to generate plots for the datasets.
    pca_plot : bool
        Whether to generate PCA plots for the datasets.
    checkpointing : bool
        Whether to check for already generated datasets during generation and skip if they are present.
    generate_small : bool
        Whether to generate the set of datasets with a smaller number of data points and simpler configurations.
    generate_standard : bool
        Whether to generate the set of datasets with a standard number of data points and moderate complexity.
    generate_dependent : bool
        Whether to generate the set of datasets with dependent features.
    multi_thread : bool
        Whether to use multi-threading for dataset generation.
    max_workers : int
        The maximum number of worker threads to use if multi_thread is True.

    Raises
    ------
    RuntimeError
        If the installed sid2regenerator version is not 0.1.1 to ensure reproducibility.

    """
    raise RuntimeError(
        'This benchmark is based on the sid2re version 0.1.1. '
        + 'To ensure reproducibility generating older benchmarks with the currently installed version '
        + 'is not allowed.',
    )


def generate_benchmark_v2(
    location: Optional[Path] = None,
    max_workers: int = 2,
) -> None:
    """Generate version v2 of the benchmarks.

    This benchmark is focused on providing an environment ablation study. Comparing the same base concept across
    different complexities, noise levels and much more.

    Parameters
    ----------
    location : Optional[Path]
        directory to which to save the generated benchmark to. If not specified the benchmark will be save to:
        ./benchmarks/benchmark_v2
    max_workers : int
        Number of workers to use to generate benchmarks. Default is set to two, number optimal workers will be
        dependent on hardware. The ProcessPoolExecutor backend is used for parallel computation.
    """
    if location is None:
        output_dir = get_script_location() / 'benchmarks' / 'benchmark_v2'
    else:
        output_dir = location / 'benchmark_v2'
    output_dir.mkdir(parents=True, exist_ok=True)
    param_grid = _load_param_grid(
        get_script_location() / 'benchmark_config_collection' / 'benchmark_v2' / 'config_grid.yaml',
    )
    print(f'▶️ Generating benchmark_v2 with {len(param_grid)} parameter combinations.')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_single_generator_for_benchmark_v2,
                config,
                idx,
                output_dir / f'{_config_hash(config)}',
            )
            for idx, config in enumerate(param_grid)
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc='Generating datasets'):  # noqa: WPS328
            pass  # noqa: WPS100,WPS420

    print(f'✔️ All datasets stored in: {output_dir}')
    _clean_benchmark_directory(output_dir, param_grid)

    index = []
    for config in param_grid:
        run_path = output_dir / f'{_config_hash(config)}'
        if (run_path / '.done').exists():
            entry = {'hash': _config_hash(config), 'path': str(run_path)}
            entry.update(_flatten_dict(config))
            index.append(entry)
    index_df = pd.DataFrame(index)
    index_df.to_csv(output_dir / 'benchmark_index.csv', index=False)
    print(f"📄 Index written to {output_dir / 'benchmark_index.csv'}")


def _run_single_generator_for_benchmark_v2(config: Dict, run_idx: int, run_path: Path) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    if (run_path / '.done').exists():
        return  # Already finished, skip
    generator = DataGeneratorGraph(
        number_of_data_points=config['number_of_data_points'],
        number_of_features=np.array(config['number_of_features']),
        number_of_outputs=config['number_of_outputs'],
        number_of_models=config['number_of_models'],
        concept_drifts=config['concept_drifts'],
        data_drifts=config['data_drifts'],
        rand_seed=config['base_seed'] + run_idx,
        noise_var=config['noise_var'],
        drift_blocking_mode=config['drift_blocking_mode'],
        continuous_time=config['continuous_time'],
        max_severity=config['max_severity'],
        min_severity=config['min_severity'],
        root_distros=config['root_distros'],
    )

    df = generator.get_data()
    concept_df, data_df = generator.get_shift_information()
    adj, names = generator.get_concept_adjacency_matrix(output_node_names=True)

    df.to_csv(run_path / 'data.csv', index=False)
    concept_df.to_csv(run_path / 'concept_drifts.csv', index=False)
    data_df.to_csv(run_path / 'data_drifts.csv', index=False)
    pd.DataFrame(adj, index=names, columns=names).to_csv(run_path / 'concept_graph.csv')

    # Save config for traceability
    with open(run_path / 'config.yaml', 'w', encoding='utf-8') as config_file:
        yaml.dump(config, config_file)
    # Marker to signal that this config has already been run
    (run_path / '.done').touch()


def _flatten_dict(dictionary: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    flattened_items: List = []
    for dict_key, dict_value in dictionary.items():
        new_key = f'{parent_key}{sep}{dict_key}' if parent_key else dict_key
        if isinstance(dict_value, dict):
            flattened_items.extend(_flatten_dict(dict_value, new_key, sep=sep).items())
        else:
            flattened_items.append((new_key, dict_value))
    return dict(flattened_items)


def _clean_benchmark_directory(output_dir: Path, param_grid: List[Dict]) -> None:
    allowed_files = {
        'data.csv',
        'concept_drifts.csv',
        'data_drifts.csv',
        'concept_graph.csv',
        'config.yaml',
        '.done',
    }
    deleted_files = 0
    deleted_folders = 0
    # First, clean extra files inside each run dir
    for run_dir in output_dir.iterdir():
        if run_dir.is_dir():
            for file in run_dir.iterdir():
                if file.name not in allowed_files:
                    file.unlink()
                    deleted_files += 1

    # Mark valid runs
    for config in param_grid:
        run_path = output_dir / f'{_config_hash(config)}'
        (run_path / '.checked').touch()

    # Now remove dirs that aren't part of the param grid
    for run_dir in output_dir.iterdir():
        if run_dir.is_dir():
            if (run_dir / '.checked').exists():
                (run_dir / '.checked').unlink()
            else:
                shutil.rmtree(run_dir)
                deleted_folders += 1

    print(f'🧹 Cleaned up {deleted_files} extraneous files and {deleted_folders} extraneous folders from {output_dir}')
