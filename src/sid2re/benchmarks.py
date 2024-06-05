# pylint: disable=unused-argument
import importlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def generate_benchmark_v1(
    location: str = "./",
    plot: bool = True,
    pca_plot: bool = True,
    checkpointing: bool = True,
    generate_small: bool = True,
    generate_standard: bool = True,
    generate_dependent: bool = True,
    multi_thread: bool = False,
    max_workers: int = 2
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

    Returns
    -------
    None
    """

    if importlib.metadata.version("sid2regenerator") != "0.1.1":
        raise RuntimeError(
            "This benchmark is based on the sid2re version 0.1.1. "
            f"To ensure reproducibility generating older benchmarks with the currently installed version "
            f"{importlib.metadata.version('sid2regenerator')} is not allowed."
        )
