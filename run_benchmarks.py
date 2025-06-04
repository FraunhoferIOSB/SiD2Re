from sid2re.benchmarks import generate_benchmark_v2

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    generate_benchmark_v2(max_workers=10)