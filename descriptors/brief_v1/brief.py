import cv2

from detectors.oriented_fast.oriented_fast import OrientedFast
from descriptors.brief_v1.pair_generator import PairGenerator


class Brief:
    def __init__(self, image, corners, s, n):
        self.image = image
        self.height, self.width = image.shape
        self.corners = corners
        self.s = s
        self.s_half = int(s / 2)
        self.n = n

    def compute(self):
        descriptors = []
        corners = []
        for c in self.corners:
            if self.__check_pixel_border(c.i, c.j):
                pos1, pos2 = PairGenerator.g1(self.s_half, self.n)
                descriptor = 0
                for k in range(self.n):
                    binary_test = 1 if self.image[c.i + pos1[k][0]][c.j + pos1[k][1]] < \
                                       self.image[c.i + pos2[k][0]][c.j + pos2[k][1]] else 0
                    descriptor += 2 ** k * binary_test
                descriptors.append(descriptor)
                corners.append(c)
        return descriptors, corners

    def __check_pixel_border(self, i, j):
        return self.s_half <= i < self.height - self.s_half or self.s_half <= j < self.width - self.s_half


def main():
    image = cv2.imread('../../resources/eagle.jpeg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = OrientedFast(grayscale_image, 3, 60, 3).detect()
    descriptors, c = Brief(grayscale_image, corners, 31, 256).compute()

    print(len(corners))
    for d in descriptors:
        print(bin(d))


if __name__ == "__main__":
    main()
