# Here all the spawners get_rate handlers

def every_ten_seconds(time: int):
    return 1 if time % 50 == 0 else 0