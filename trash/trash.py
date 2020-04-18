from descriptors.brief_v1.pair_generator import PairGenerator
import cv2

image = cv2.imread('../resources/eagle.jpeg', 0)
n = 12
pos1, pos2 = PairGenerator.g1(15, n)
descriptor = ""
d = 0
dd = 0

for k in range(n):
    binary_test = "1" if image[25 + pos1[k][0]][33 + pos1[k][1]] < \
                         image[25 + pos2[k][0]][33 + pos2[k][1]] else "0"
    bt = 1 if image[25 + pos1[k][0]][33 + pos1[k][1]] < image[25 + pos2[k][0]][33 + pos2[k][1]] else 0
    descriptor += binary_test
    d = d * 10 + bt
    dd += 2 ** k * bt

print(descriptor)
print(d)
print(dd, bin(dd))
# print(pos1)
# print(pos2)
