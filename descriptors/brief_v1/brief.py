import cv2
import numpy as np

from descriptors.brief_v1.pair_generator import PairGenerator
from detectors.oriented_fast.oriented_fast import OrientedFast


class Brief:
    def __init__(self, image, corners, s, n):
        self.image = image
        self.height, self.width = image.shape
        self.corners = corners
        self.s = s
        self.s_half = int(s / 2)
        self.n = n
        self.__init_rotation_matrices()

    def __init_rotation_matrices(self):
        self.sector = 12
        alpha = [i for i in range(0, self.sector * 31, self.sector)]
        self.matrices = [cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=1) for a in alpha]

    def compute(self):
        descriptors = []
        corners = []
        for c in self.corners:
            if self.__check_pixel_border(c.i, c.j):
                pos1, pos2 = PairGenerator.g1(self.s_half, self.n)

                index = round(c.angle / self.sector)
                rotation_matrix = self.matrices[index]

                new_pos1 = np.stack([pos1[:, 1], pos1[:, 0], np.zeros(len(pos1))], axis=1)
                new_pos1 = np.round(np.dot(new_pos1, rotation_matrix.T))
                new_pos1 = new_pos1.astype(int)
                new_pos2 = np.stack([pos2[:, 1], pos2[:, 0], np.zeros(len(pos2))], axis=1)
                new_pos2 = np.round(np.dot(new_pos2, rotation_matrix.T))
                new_pos2 = new_pos2.astype(int)

                descriptor = 0
                for k in range(self.n):
                    binary_test = 1 if self.image[c.i + new_pos1[k][0]][c.j + new_pos1[k][1]] < \
                                       self.image[c.i + new_pos2[k][0]][c.j + new_pos2[k][1]] else 0
                    descriptor += 2 ** k * binary_test
                descriptors.append(descriptor)
                corners.append(c)
        return descriptors, corners

    def __check_pixel_border(self, i, j):
        return self.s_half <= i < self.height - self.s_half or self.s_half <= j < self.width - self.s_half


def main():
    image = cv2.imread('../../resources/eagle.jpeg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = OrientedFast(grayscale_image, 3, 60, 3, "x").detect()
    descriptors, c = Brief(grayscale_image, corners, 31, 256).compute()


if __name__ == "__main__":
    main()
