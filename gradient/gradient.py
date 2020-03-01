import numpy as np

from convolution.convolution import convolution_pixel


def calculate_gradient(image, mask_x, mask_y):
    height, width = image.shape

    dx = np.zeros((height, width), dtype=float)
    dy = np.zeros((height, width), dtype=float)

    for i in range(height):
        for j in range(width):
            dx[i][j] = convolution_pixel(image, mask_x, i, j)
            dy[i][j] = convolution_pixel(image, mask_y, i, j)

    return dx, dy
