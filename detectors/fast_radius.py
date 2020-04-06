import cv2

from utils.image_shower import show_image
from utils.masks_generator import *


class Fast:
    def __init__(self, image, radius, threshold):
        self.image = image
        self.height, self.width = image.shape
        self.radius = radius
        self.threshold = threshold
        self.__init_masks(radius)
        self.check_coefficient = 0.75

    def __init_masks(self, r):
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        masks = (x ** 2 + y ** 2 >= r ** 2 - 1) * (x ** 2 + y ** 2 <= r ** 2 + 1)
        j, i = np.where(masks)
        points_count = len(i)
        left = []
        right = []
        first = []
        second = []
        third = []
        fourth = []
        for k in range(points_count):
            if j[k] == 0:
                left.append(k)
            elif j[k] == 2 * r:
                right.append(k)
            elif j[k] >= r > i[k]:
                first.append(k)
            elif j[k] > r <= i[k]:
                second.append(k)
            elif j[k] <= r < i[k]:
                third.append(k)
            elif j[k] < r >= i[k]:
                fourth.append(k)
        indices = first + right + second[::-1] + third[::-1] + left[::-1] + fourth
        self.alpha_i = []
        self.alpha_j = []
        for k in range(points_count):
            self.alpha_i.append(i[indices[k]] - r)
            self.alpha_j.append(j[indices[k]] - r)
        self.non_fast_indices = np.arange(0, points_count)
        self.fast_indices = [0, points_count // 4, points_count // 2, 3 * points_count // 4]

    def detect(self):
        result = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                if self.__check(i, j, True):
                    if self.__check(i, j, False):
                        result[i][j] = 1
        return result

    def __check(self, current_i, current_j, is_fast_check):
        first_condition = self.image[current_i][current_j] + self.threshold
        second_condition = self.image[current_i][current_j] - self.threshold
        first_condition_count = 0
        second_condition_count = 0
        indices = self.non_fast_indices
        if is_fast_check:
            indices = self.fast_indices
        for k in range(len(indices)):
            i = self.alpha_i[indices[k]]
            j = self.alpha_j[indices[k]]
            current_value = self.__get_value(i + current_i, j + current_j)
            if current_value > first_condition:
                first_condition_count += 1
            else:
                first_condition_count = 0
            if current_value < second_condition:
                second_condition_count += 1
            else:
                second_condition_count = 0

        result_condition = self.check_coefficient * len(indices)
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

    result = Fast(grayscale_image, 3, 60).detect()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if result[i][j] > 0:
                cv2.circle(image, (j, i), 2, (0, 0, 255), 1)

    show_image(image)


main()
