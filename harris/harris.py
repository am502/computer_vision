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
            if r > threshold:
                result[i][j] = r

    result = nms(result, 9)
    # result[result > threshold] = 255
    return result


def main():
    grayscale_image = cv2.imread('waffle.jpg', 0)

    grayscale_image = convolution(grayscale_image, get_gauss_kernel(5, 1.4))

    mask_x, mask_y = get_lucas_kanade_masks()
    dx, dy = calculate_gradient(grayscale_image, mask_x, mask_y)

    result = harris(grayscale_image, dx, dy, get_gauss_kernel(5, 1.4), 0.04, 0.4)
    show_image(result)


main()
