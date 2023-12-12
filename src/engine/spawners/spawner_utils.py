# Here all the spawners get_rate handlers

def every_ten_seconds(time: int):
    return 10 if time % 20 == 0 else 0