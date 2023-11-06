from ..constants import ACCELERATION

def carSpeed(speedLimit, currentSpeed):
    return min(speedLimit, currentSpeed + ACCELERATION)
