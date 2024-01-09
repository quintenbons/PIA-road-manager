import math

def getLength(pos1:tuple[float, float], pos2:tuple[float, float]) -> float:
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def circle_collision(pos1, pos2, size1, size2):
    dist = getLength(pos1, pos2)
    if dist < (size1 + size2):
        return True
    return False

def norm(v) -> float:
    return (v[0]**2+v[1]**2)**0.5

def vecteur(pos1, pos2):
    return (pos2[0] - pos1[0], pos2[1] - pos1[1])

def vecteur_norm(pos1, pos2):
    vec = vecteur(pos1, pos2)
    n = norm(vec)
    return (vec[0]/n, vec[1]/n)

def scalaire(u, v):
    return u[0]*v[0] + u[1]*v[1]

def get_angle(center:tuple[float, float], point:tuple[float, float]) -> float:
    x, y = point[0] - center[0], point[1] - center[1]
    angle_rad = math.atan2(y, x)
    return angle_rad