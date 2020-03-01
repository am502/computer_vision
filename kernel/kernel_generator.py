import math

import numpy as np


def get_box_kernel(size):
    return np.ones((size, size)) / size / size


def get_gauss_kernel(radius, sigma):
    bound = int(radius / 2)
    kernel = np.zeros((radius, radius), dtype=float)
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            kernel[i + bound][j + bound] = __gauss_function(i, j, sigma)
    return kernel / kernel.sum()


def __gauss_function(x, y, sigma):
    return math.e ** ((- x * x - y * y) / 2 / sigma / sigma) / math.sqrt(2 * math.pi) / sigma


def get_lk_x_kernel():
    return np.array([[-1, 8, 0, -8, 1]]) / 12


def get_lk_y_kernel():
    return get_lk_x_kernel().T


def get_sobel_x_kernel():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def get_sobel_y_kernel():
    return get_sobel_x_kernel().T
