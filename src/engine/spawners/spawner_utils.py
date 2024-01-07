# Here all the spawners get_rate handlers

from engine.constants import TIME

def benchmark_spawner(time: float):
    return 1 if time % 5 < TIME else 0

def benchmark_spawner2(time: float):
    return 1 if time % 10 < TIME else 0

def every_ten_seconds(time: float):
    return 1 if time % 4 < TIME else 0

def spawn_fast(time: float):
    return 1 if time % 2 < TIME else 0

def spawn_medium(time: float):
    return 1 if time % 4 < TIME else 0

def spawn_slow(time: float):
    return 1 if time % 8 < TIME else 0

def spawner_frequencies(handler):
    return {
        "benchmark_spawner": benchmark_spawner,
        "benchmark_spawner2": benchmark_spawner2,
        "uniform_spawner": every_ten_seconds,
        "fast": spawn_fast,
        "medium": spawn_medium,
        "slow": spawn_slow
    }[handler]
