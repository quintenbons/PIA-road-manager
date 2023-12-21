# Here all the spawners get_rate handlers

# Benchmark 1
def benchmark_spawner(time: int):
    return 1 if time % 5 == 0 else 0

# Benchmark 2
def benchmark_spawner2(time: int):
    return 1 if time % 10 == 0 else 0

def every_ten_seconds(time: int):
    return 1 if time % 4 == 0 else 0

def spawner_handlers(handler):
    return {
        "benchmark_spawner": benchmark_spawner,
        "benchmark_spawner2": benchmark_spawner2,
        "every_ten_seconds": every_ten_seconds
    }[handler]
