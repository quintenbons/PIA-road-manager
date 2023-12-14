# Here all the spawners get_rate handlers

def benchmark_spawner(time: int):
    return 1 if time % 5 == 0 else 0

def every_ten_seconds(time: int):
    return 10 if time % 20 == 0 else 0