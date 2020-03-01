import cv2

from convolution.convolution import convolution
from kernel.kernel_generator import *
from utils.image_shower import show_image
from utils.nms import nms


def harris(image, dx, dy, w, k, threshold):
    i_x = convolution(image, dx)
    i_y = convolution(image, dy)

    s_2_x = convolution(i_x * i_x, w)
    print(1)
    s_2_y = convolution(i_y * i_y, w)
    print(2)
    s_x_y = convolution(i_x * i_y, w)
    print(3)

    height, width = image.shape
    result = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            det = s_2_x[i][j] * s_2_y[i][j] - s_x_y[i][j] * s_x_y[i][j]
            trace = s_2_x[i][j] * s_2_x[i][j] + s_2_y[i][j] * s_2_y[i][j]
            r = det - k * trace * trace
            result[i][j] = r
    res = nms(result, 9)
    image[res > threshold] = 255
    return image


grayscale_image = cv2.imread('house.jpg', 0)

# dst = cv2.cornerHarris(grayscale_image,2,3,0.04)
# height, width = grayscale_image.shape
# for i in range(height):
#     for j in range(width):
#         if dst[i][j] > 1:
#             print(dst[i][j])
# dst = cv2.dilate(dst, None)
# grayscale_image[dst > 0.01 * dst.max()] = 255
# show_image(grayscale_image)

show_image(harris(grayscale_image, get_lk_x_kernel(), get_lk_y_kernel(), get_box_kernel(5), 0.04, 0.4))
