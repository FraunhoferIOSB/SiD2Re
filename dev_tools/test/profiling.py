import os
import pstats
from pstats import SortKey
from pathlib import Path


def sort_and_print_profiling_data(dir):
    files_to_analyse = os.listdir(dir)
    files_to_analyse.remove("combined.prof")
    p = pstats.Stats(*[str(Path(dir) / file) for file in files_to_analyse])
    p.dump_stats(Path(dir)/"combined.prof")
    p = pstats.Stats(str(Path(dir) / "combined.prof"))
    p.sort_stats(SortKey.CUMULATIVE).print_stats("tadsim", 15)
    p.sort_stats('tottime').print_stats("tadsim", 15)
    for file in files_to_analyse:
        os.remove(f"prof/{file}")


if __name__ == "__main__":
    sort_and_print_profiling_data("prof")
