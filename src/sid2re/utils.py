from pathlib import Path


# Seaborn visualization library


def get_script_location() -> Path:
    """
    Gets the root directory of the project by resolving the known path relative to this file.

    Returns
    -------
    Path
        The project root directory.
    """

    return Path(__file__)
