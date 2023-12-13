from ..constants import ACCELERATION, TIME

def car_speed(speedLimit, currentSpeed, acceleration):
    return max(min(speedLimit, currentSpeed + acceleration*TIME), 0)

def car_position(road_len, current_pos, speed):
    return min(road_len, current_pos + speed*TIME)

def mov_node_position(pos, pos_vec, speed, vec):
    ds = speed*TIME
    new_pos_vec = (pos_vec[0] + vec[0] * ds, pos_vec[1] + vec[1]*ds)
    new_pos = pos + ds

    return new_pos, new_pos_vec
