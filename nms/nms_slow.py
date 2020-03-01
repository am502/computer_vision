def nms(image, radius):
    bound = int(radius / 2)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            current_max = __get_max(image, height, width, bound, i, j)
            if image[i][j] != current_max:
                image[i][j] = 0
            else:
                image[i][j] = 255

    return image


def __get_max(image, height, width, bound, current_i, current_j):
    current_max = __get_value(image, height, width, current_i, current_j)
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            current_value = __get_value(image, height, width, i + current_i, j + current_j)
            if current_value > current_max:
                current_max = current_value
    return current_max


def __get_value(image, height, width, i, j):
    if i < 0:
        i = i - 1
    if j < 0:
        j = -j - 1
    if i >= height:
        i = height - (i - height) - 1
    if j >= width:
        j = width - (j - width) - 1
    return image[i][j]
