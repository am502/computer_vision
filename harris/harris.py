import cv2

from convolution.convolution import convolution
from nms.nms_slow import nms
from utils.image_shower import show_image
from utils.kernel_generator import *
from utils.masks_generator import *


def harris(height, width, dx, dy, w, k):
    s_2_x = convolution(dx * dx, w)
    s_2_y = convolution(dy * dy, w)
    s_x_y = convolution(dx * dy, w)

    r = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            det = s_2_x[i][j] * s_2_y[i][j] - s_x_y[i][j] * s_x_y[i][j]
            trace = s_2_x[i][j] + s_2_y[i][j]
            r[i][j] = det - k * trace * trace

    return nms(r, 9)


def main():
    image = cv2.imread('eagle.jpeg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape

    mask_x, mask_y = get_sobel_masks()
    dx = convolution(grayscale_image, mask_x)
    dy = convolution(grayscale_image, mask_y)

    result = harris(height, width, dx, dy, get_gauss_kernel(5, 1.4), 0.04)

    for i in range(height):
        for j in range(width):
            if result[i, j] > 0.1 * result.max():
                cv2.circle(image, (j, i), 2, (0, 0, 255), 1)

    show_image(result, image)


main()
