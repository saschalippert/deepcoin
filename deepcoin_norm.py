import numpy as np

class Normalizer_Noop:

    def __init__(self):
        pass

    def normalize(self, data):
        return data

    def denormalize(self, data):
        return data

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

class Normalizer_Min_Max_Target:

    def __init__(self, target_min, target_max):
        self.target_min = target_min
        self.target_max = target_max
        pass

    def normalize(self, data):
        self.data_max = np.max(data)
        self.data_min = np.min(data)
        self.data_scale = self.data_max - self.data_min

        self.targe_scale = self.target_max - self.target_min

        return ((data - self.data_min) / self.data_scale)  * self.targe_scale + self.target_min

    def denormalize(self, data):
        return ((data - self.target_min) / self.targe_scale * self.data_scale) + self.data_min

class Normalizer_ClipStdDev:

    def __init__(self):
        pass

    def normalize(self, data):
        self.mean = np.mean(data)
        self.stddev = np.std(data)

        self.min = self.mean - (5 * self.stddev)
        self.max = self.mean + (5 * self.stddev)
        self.scale = (self.max - self.min) / 2

        clipped = np.clip(
            data,
            self.min,
            self.max
        )

        return ((clipped - self.min) / self.scale) - 1

    def denormalize(self, data):
        return ((1 + data) * self.scale) + self.min
