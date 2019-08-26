import numpy as np


def norm_min_max(data):
    data_max = np.max(data)
    data_min = np.min(data)
    scale = data_max - data_min

    return (data - data_min) / scale
