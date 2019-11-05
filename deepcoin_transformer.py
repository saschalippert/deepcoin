import numpy as np


class Transformer_Noop:

    def __init__(self):
        pass

    def transform(self, data):
        return data

    def revert_single(self, start, data):
        return data

    def revert_list(self, start, data):
        return data


class Transformer_SimpleReturn:

    def __init__(self):
        pass

    def transform(self, data):
        return (data[1:] / data[0:-1]) - 1

    def revert_single(self, start, data):
        return start * (data + 1)

    def revert_list(self, start, data):
        result = np.zeros(len(data) + 1)
        result[0] = start

        for i in range(1, len(result)):
            result[i] = self.revert_single(result[i - 1], data[i - 1])

        return result

class Transformer_LogReturn:

    def __init__(self):
        pass

    def transform(self, data):
        return np.log(data[1:] / data[0:-1])

    def revert_single(self, start, data):
        return start * np.exp(data)

    def revert_list(self, start, data):
        result = np.zeros(len(data) + 1)
        result[0] = start

        for i in range(1, len(result)):
            result[i] = self.revert_single(result[i - 1], data[i - 1])

        return result
