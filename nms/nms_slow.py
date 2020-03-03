import numpy as np


def nms_slow(image, radius):
    bound = int(radius / 2)
    height, width = image.shape
    processed_image = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            processed_image[i][j] = __check_local_max(image, height, width, bound, i, j)
    return processed_image


def __check_local_max(image, height, width, bound, current_i, current_j):
    current_value = __get_value(image, height, width, current_i, current_j)
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            if __get_value(image, height, width, i + current_i, j + current_j) > current_value:
                return 0
    return current_value


def __get_value(image, height, width, i, j):
    if i < 0:
        i = i - 1
    if j < 0:
        j = -j - 1
    if i >= height:
        i = height - (i - height) - 1
    if j >= width:
        j = width - (j - width) - 1
    return image[i][j]
