import cv2

from convolution.convolution import convolution
from nms.nms_slow import nms_slow
from utils.image_shower import show_image
from utils.kernel_generator import *
from utils.masks_generator import *


def harris(grayscale_image, mask_x, mask_y, w, k):
    dx = convolution(grayscale_image, mask_x)
    dy = convolution(grayscale_image, mask_y)

    s_2_x = convolution(dx * dx, w)
    s_2_y = convolution(dy * dy, w)
    s_x_y = convolution(dx * dy, w)

    height, width = grayscale_image.shape
    r = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            det = s_2_x[i][j] * s_2_y[i][j] - s_x_y[i][j] * s_x_y[i][j]
            trace = s_2_x[i][j] + s_2_y[i][j]
            r[i][j] = det - k * trace * trace
    return nms_slow(r, 9)


def main():
    image = cv2.imread('../resources/house.jpg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape

    mask_x, mask_y = get_lucas_kanade_masks()

    responses = harris(grayscale_image, mask_x, mask_y, get_gauss_kernel(5, 1.4), 0.04)

    for i in range(height):
        for j in range(width):
            if responses[i][j] > 0.1 * responses.max():
                cv2.circle(image, (j, i), 2, (0, 0, 255), 1)

    show_image(responses, image)


main()
