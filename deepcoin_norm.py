import numpy as np


class Normalizer_Min_Max:

    def __init__(self):
        pass

    def normalize(self, data):
        self.data_max = np.max(data)
        self.data_min = np.min(data)
        self.scale = self.data_max - self.data_min

        return (data - self.data_min) / self.scale

    def denormalize(self, data):
        return (data * self.scale) + self.data_min

class Normalizer_Min_Max2:

    def __init__(self):
        pass

    def normalize(self, data):
        self.data_max = np.max(data)
        self.data_min = np.min(data)
        self.scale = self.data_max - self.data_min

        return ((data - self.data_min) / (self.scale / 2)) - 1

    def denormalize(self, data):
        return 1
