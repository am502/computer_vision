import math

import cv2

from convolution.convolution import convolution_pixel, convolution
from gradient.gradient import calculate_gradient
from nms.nms_slow import nms
from utils.image_shower import show_image
from utils.kernel_generator import *
from utils.masks_generator import *


def harris(image, dx, dy, w, k, threshold):
    i_2_x = dx * dx
    i_2_y = dy * dy
    i_x_y = dx * dy

    height, width = image.shape
    result = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            s_2_x = convolution_pixel(i_2_x, w, i, j)
            s_2_y = convolution_pixel(i_2_y, w, i, j)
            s_x_y = convolution_pixel(i_x_y, w, i, j)

            det = s_2_x * s_2_y - s_x_y * s_x_y
            trace = s_2_x * s_2_x + s_2_y * s_2_y
            r = det - k * trace * trace

            result[i][j] = r

    res = nms(result, 9)
    res[res > threshold] = 255
    return res


def main():
    grayscale_image = cv2.imread('house.jpg', 0)

    grayscale_image = convolution(grayscale_image, get_gauss_kernel(5, 0.66))

    mask_x, mask_y = get_lucas_kanade_masks()
    dx, dy = calculate_gradient(grayscale_image, mask_x, mask_y)

    height, width = grayscale_image.shape
    magnitudes = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            magnitudes[i][j] = math.sqrt(dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j])
    show_image(magnitudes)

    result = harris(grayscale_image, dx, dy, get_gauss_kernel(5, 0.66), 0.04, 0.4)
    show_image(result)


main()
