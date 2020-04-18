import numpy as np


class PairGenerator:
    @staticmethod
    def g1(s, n):
        pos1 = np.random.randint(-s, s + 1, size=(n, 2))
        pos2 = np.random.randint(-s, s + 1, size=(n, 2))
        return pos1, pos2
