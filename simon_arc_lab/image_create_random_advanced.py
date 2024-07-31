import random
from .image_util import *
from .image_create_random_simple import *

def image_create_random_advanced(seed, min_width, max_width, min_height, max_height):
    """
    Generate a random image.

    :param seed: The seed for the random number generator
    :param max_image_size: The maximum size of the image
    :return: image
    """

    width = random.Random(seed + 1).randint(min_width, max_width)
    height = random.Random(seed + 2).randint(min_height, max_height)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    color3 = colors[3]

    image_types = ['one_color', 'two_colors', 'three_colors', 'four_colors']
    image_type = random.Random(seed + 4).choice(image_types)

    image = None
    if image_type == 'one_color':
        image = image_create(width, height, color0)
    if image_type == 'two_colors':
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        ratio = random.Random(seed + 5).choice(ratios)
        image = image_create_random_with_two_colors(width, height, color0, color1, ratio, seed + 6)
    if image_type == 'three_colors':
        weights = [1, 1, 1, 2, 3, 4, 7, 11]
        random.Random(seed + 5).shuffle(weights)
        weight0 = weights[0]
        weight1 = weights[1]
        weight2 = weights[2]
        image = image_create_random_with_three_colors(width, height, color0, color1, color2, weight0, weight1, weight2, seed + 10)
    if image_type == 'four_colors':
        weights = [1, 1, 1, 1, 2, 2, 3, 3, 4, 7, 11]
        random.Random(seed + 5).shuffle(weights)
        weight0 = weights[0]
        weight1 = weights[1]
        weight2 = weights[2]
        weight3 = weights[3]
        image = image_create_random_with_four_colors(width, height, color0, color1, color2, color3, weight0, weight1, weight2, weight3, seed + 10)

    number_of_lines = random.Random(seed + 6).randint(0, 9)
    for i in range(number_of_lines):
        x0 = random.Random(seed + 7 + i).randint(0, width - 1)
        x1 = random.Random(seed + 8 + i).randint(0, width - 1)
        y0 = random.Random(seed + 9 + i).randint(0, height - 1)
        y1 = random.Random(seed + 10 + i).randint(0, height - 1)
        color = random.Random(seed + 11 + i).choice(colors)
        image = bresenham_line(image, x0, y0, x1, y1, color)

    return image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 1
    for i in range(n):
        image = image_create_random_advanced(i, 10, 20, 10, 20)
        plt.imshow(image, cmap='gray')
        plt.show()

