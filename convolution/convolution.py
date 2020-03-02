import numpy as np


def convolution(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    bound_i = int(kernel_height / 2)
    bound_j = int(kernel_width / 2)
    processed_image = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            processed_image[i][j] = __calculate_value(image, height, width, kernel, bound_i, bound_j, i, j)
    return processed_image


def __calculate_value(image, height, width, kernel, bound_i, bound_j, current_i, current_j):
    value = 0
    for i in range(-bound_i, bound_i + 1):
        for j in range(-bound_j, bound_j + 1):
            value += kernel[i + bound_i][j + bound_j] * __get_value(image, height, width, i + current_i, j + current_j)
    return value


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
