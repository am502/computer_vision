import cv2

from utils.image_shower import show_image
from utils.masks_generator import *


class Fast:
    alpha_i = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3]
    alpha_j = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
    fast_indices = [0, 4, 8, 12]
    check_coefficient = 0.75

    def __init__(self, image, threshold):
        self.image = image
        self.height, self.width = image.shape
        self.threshold = threshold

    def detect(self):
        result = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                if self.__fast_check(i, j):
                    if self.__check(i, j):
                        result[i][j] = 1
        return result

    def __fast_check(self, current_i, current_j):
        first_condition = self.image[current_i, current_j] + self.threshold
        second_condition = self.image[current_i, current_j] - self.threshold
        first_condition_count = 0
        second_condition_count = 0
        for k in range(len(Fast.fast_indices)):
            i = Fast.alpha_i[Fast.fast_indices[k]]
            j = Fast.alpha_j[Fast.fast_indices[k]]
            current_value = self.__get_value(i + current_i, j + current_j)
            if current_value > first_condition:
                first_condition_count += 1
            if current_value < second_condition:
                second_condition_count += 1

        result_condition = Fast.check_coefficient * len(Fast.fast_indices)
        if first_condition_count >= result_condition or second_condition_count >= result_condition:
            return True
        return False

    def __check(self, current_i, current_j):
        first_condition = self.image[current_i, current_j] + self.threshold
        second_condition = self.image[current_i, current_j] - self.threshold
        first_condition_count = 0
        second_condition_count = 0
        for k in range(len(Fast.alpha_i)):
            i = Fast.alpha_i[k]
            j = Fast.alpha_j[k]
            current_value = self.__get_value(i + current_i, j + current_j)
            if current_value > first_condition:
                first_condition_count += 1
            else:
                first_condition_count = 0
            if current_value < second_condition:
                second_condition_count += 1
            else:
                second_condition_count = 0

        result_condition = Fast.check_coefficient * len(Fast.alpha_i)
        if first_condition_count >= result_condition or second_condition_count >= result_condition:
            return True
        return False

    def __get_value(self, i, j):
        if i < 0:
            i = i - 1
        if j < 0:
            j = -j - 1
        if i >= self.height:
            i = self.height - (i - self.height) - 1
        if j >= self.width:
            j = self.width - (j - self.width) - 1
        return self.image[i][j]


def main():
    image = cv2.imread('../resources/house.jpg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = Fast(grayscale_image, 60).detect()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if result[i][j] > 0:
                cv2.circle(image, (j, i), 2, (0, 0, 255), 1)

    show_image(image)


main()
