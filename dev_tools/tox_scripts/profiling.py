import pstats
from pstats import SortKey

def sort_and_print_profiling_data(file):
    p = pstats.Stats(file)
    p.sort_stats(SortKey.CUMULATIVE).print_stats("src",15)
    p.sort_stats(SortKey.TIME).print_stats("src",15)

if __name__ == "__main__":
    sort_and_print_profiling_data("prof/combined.prof")
