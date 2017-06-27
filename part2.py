from PIL import Image
import numpy as np
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from scipy.signal import medfilt
import skimage.filters as filters

# Transition function for denoising the image
def transition(img, x, y):
    if x == img.shape[1] - 1 or x == 0 or y == img.shape[0] - 1 or y == 0:
        return


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


def mark_neighbourhoods(input, initial_threshold):
    marked = np.zeros(input.shape)

    threshold = threshold_otsu(input)

    diff = threshold - initial_threshold

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            neighbourhood = moore_neighbourhood(input, j, i, 2)     
            count = 0
            for colour in neighbourhood:
                if colour < threshold:
                    count += 1
            average = count / len(neighbourhood)
            percentage = average * 100

            #if diff < percentage < threshold:
            #    marked[i][j] = 255

            if diff < percentage < 60:
                marked[i][j] = 255

    return marked


def denoise_gaussian(img):
    return filters.gaussian(img, sigma=2)


def denoise_median(input):
    return filters.median(input, selem=disk(15))


def denoise_mean(input):
    denoised = np.copy(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            neighbourhood = moore_neighbourhood(input, j, i, 1)
            sum = 0
            for val in neighbourhood:
                sum += val
            avg = sum / len(neighbourhood)
            denoised[i][j] = avg

    return denoised
