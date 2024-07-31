import random
from .image_util import *

def image_create_random_with_two_colors(width: int, height: int, color1: int, color2: int, ratio: float, seed: int) -> np.array:
    image = image_create(width, height, color1)

    positions = []
    for y in range(height):
        for x in range(width):
            positions += [(y, x)]

    random.Random(seed).shuffle(positions)

    # take a ratio of the positions
    num_positions = int(len(positions) * ratio)
    for i in range(num_positions):
        y, x = positions[i]
        image[y, x] = color2
    return image

def image_create_random_with_three_colors(width: int, height: int, color0: int, color1: int, color2: int, weight0: int, weight1: int, weight2: int, seed: int) -> np.array:
    image = image_create(width, height, color0)

    positions = [(y, x) for y in range(height) for x in range(width)]

    random.Random(seed).shuffle(positions)

    total_weight = weight0 + weight1 + weight2
    num_positions_a = int(len(positions) * (weight1 / total_weight))
    num_positions_b = int(len(positions) * (weight2 / total_weight))

    for i in range(num_positions_a):
        y, x = positions[i]
        image[y, x] = color1

    for i in range(num_positions_b):
        y, x = positions[i + num_positions_a]
        image[y, x] = color2

    return image

def image_create_random_with_four_colors(width: int, height: int, color0: int, color1: int, color2: int, color3: int, weight0: int, weight1: int, weight2: int, weight3: int, seed: int) -> np.array:
    image = image_create(width, height, color0)

    positions = [(y, x) for y in range(height) for x in range(width)]

    random.Random(seed).shuffle(positions)

    total_weight = weight0 + weight1 + weight2 + weight3
    num_positions_a = int(len(positions) * (weight1 / total_weight))
    num_positions_b = int(len(positions) * (weight2 / total_weight))
    num_positions_c = int(len(positions) * (weight3 / total_weight))

    for i in range(num_positions_a):
        y, x = positions[i]
        image[y, x] = color1

    for i in range(num_positions_b):
        y, x = positions[i + num_positions_a]
        image[y, x] = color2

    for i in range(num_positions_c):
        y, x = positions[i + num_positions_a + num_positions_b]
        image[y, x] = color3

    return image
