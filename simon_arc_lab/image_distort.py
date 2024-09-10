import numpy as np
import random

def image_distort(image: np.array, number_of_iterations: int, impact_ratio: int, seed: int) -> np.array:
    """
    Distort the image by swapping adjacent pixels over and over.

    :param image: The image to distort
    :param number_of_iterations: The number of iterations to swap pixels.
    :param impact_ratio: The percentage of pixels to swap
    :param seed: The seed for the random number generator
    :return: The distorted image
    """
    if impact_ratio < 0 or impact_ratio > 100:
        raise Exception("impact_ratio must be between 0 and 100.")
    if number_of_iterations < 1:
        raise Exception("number_of_iterations must be 1 or greater.")
    if number_of_iterations > 256:
        raise Exception("number_of_iterations is beyond what makes sense.")
    
    height, width = image.shape
    all_swap_x_positions = []
    for y in range(height):
        for x in range(width-1):
            all_swap_x_positions.append((x, y))
    all_swap_y_positions = []
    for x in range(width):
        for y in range(height-1):
            all_swap_y_positions.append((x, y))
    result = image.copy()
    for i in range(number_of_iterations):
        # shuffle the positions
        random.Random(seed + i * 1000 + 1).shuffle(all_swap_x_positions)
        random.Random(seed + i * 1000 + 2).shuffle(all_swap_y_positions)

        # take N percent of the positions
        swap_x_positions = all_swap_x_positions[:len(all_swap_x_positions) * impact_ratio // 100]
        swap_y_positions = all_swap_y_positions[:len(all_swap_y_positions) * impact_ratio // 100]

        # horizontal swap adjacent pixels
        for x, y in swap_x_positions:
            a = result[y, x]
            b = result[y, x + 1]
            result[y, x] = b
            result[y, x + 1] = a

        # vertical swap adjacent pixels
        for x, y in swap_y_positions:
            a = result[y, x]
            b = result[y + 1, x]
            result[y, x] = b
            result[y + 1, x] = a
    return result
