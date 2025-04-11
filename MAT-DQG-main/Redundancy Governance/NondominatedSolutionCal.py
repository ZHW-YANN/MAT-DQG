import numpy as np


# 非支配解的计算
class NonSolu(object):
    def __init__(self, size, data, min_values):
        self.size = size
        self.data = data
        self.min_values = min_values

    def change_values(self, value, min_value_index):
        new_value = 0.1 ** np.sign(value - self.min_values[min_value_index]) * np.abs(
            value - self.min_values[min_value_index])
        return new_value

    def find_nondomination(self):
        num_of_dominated = np.zeros(self.size)
        sets_of_domination = [[] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if self.change_values(self.data[i]['value1'], 0) <= self.change_values(self.data[j]['value1'], 0) \
                        and self.change_values(self.data[i]['value2'], 1) <= self.change_values(self.data[j]['value2'], 1):
                    if self.change_values(self.data[i]['value1'], 0) < self.change_values(self.data[j]['value1'], 0) \
                            or self.change_values(self.data[i]['value2'], 1) < self.change_values(self.data[j]['value2'], 1):
                        num_of_dominated[j] += 1
                        sets_of_domination[i].append(j)
        return num_of_dominated, sets_of_domination
