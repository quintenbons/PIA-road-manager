from ..constants import ACCELERATION, TIME

def car_speed(speedLimit, currentSpeed, acceleration):
    return max(min(speedLimit, currentSpeed + acceleration*TIME), 0)

def car_position(road_len, current_pos, speed):
    return min(road_len, current_pos + speed*TIME)