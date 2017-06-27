from PIL import Image
import numpy as np


def transition(img, x, y):
    val = img[y][x]

    neighbourhood = von_neumann(img, x, y, 1)

    for neighbour in neighbourhood:
        if abs(int(val) - int(neighbour)) >= epsilon:
            return False

    return True


def ca_edge(img, threshold):
    global epsilon
    global new_img
    epsilon = threshold
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            change = transition(img, j, i)
            if change:
                new_img[i][j] = 0
            else:
                new_img[i][j] = img[i][j]

    return new_img


def denoise_mode(input):
    denoised = np.copy(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            neighbourhood = moore_neighbourhood(input, j, i, 3)
            max_occ = 0
            max_val = 0
            for val in neighbourhood:
                num = neighbourhood.count(val)
                if (num > max_occ):
                    max_occ = num
                    max_val = val
            if max_occ == 1:
                denoised[i][j] = input[i][j]
            else:
                denoised[i][j] = max_val

    return denoised


def moore_neighbourhood(input, x, y, radius):
    neighbours = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i == 0 and j == 0:
                continue
            if y + i < 0 or y + i > input.shape[0] - 1 or x + j < 0 or x + j > input.shape[1] - 1:
                continue
            neighbour = input[y + i][x + j]
            neighbours += [neighbour]

    return neighbours


def von_neumann(input, x, y, radius):
    neighbours = []
    for i in range(2):
        for j in range(-radius, radius + 1):
            if j == 0:
                continue
            if i == 0:
                if x + j > input.shape[1] - 1 or x + j < 0:
                    continue
                neighbour = input[y][x + j]
            else:
                if y + j > input.shape[0] - 1 or y + j < 0:
                    continue
                neighbour = input[y + j][x]
            neighbours += [neighbour]

    return neighbours
