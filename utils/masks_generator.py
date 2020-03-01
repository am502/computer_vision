import numpy as np


def get_lucas_kanade_masks():
    mask_x = np.array([[-1, 8, 0, -8, 1]]) / 12
    mask_y = mask_x.T
    return mask_x, mask_y


def get_sobel_masks():
    mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask_y = mask_x.T
    return mask_x, mask_y
