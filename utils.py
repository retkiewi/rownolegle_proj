import numpy as np
from math import sqrt

def get_distance(pos_a: np.ndarray, pos_b: np.ndarray):
    return  sqrt(sum([d**2 for d in pos_a - pos_b]))