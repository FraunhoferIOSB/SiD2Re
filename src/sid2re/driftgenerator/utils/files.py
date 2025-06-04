"""Basic utilities for working with config, loading and saving."""
from pathlib import Path


def get_script_location() -> Path:
    """
    Get the root directory of the project by resolving the known path relative to this file.

    Returns
    -------
    Path
        The project root directory.
    """
    return Path(__file__).parent.parent.parent
