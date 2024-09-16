import numpy as np
from .histogram import *

def image_vote(images: list[np.array]) -> np.array:
    size_counters = {}
    for image in images:
        size = image.shape
        size_counters[size] = size_counters.get(size, 0) + 1
    # Find the highest counters
    max_counter = 0
    for size, counter in size_counters.items():
        if counter > max_counter:
            max_counter = counter

    # Get rid of the size_counters that are not the max_counter
    for size in list(size_counters.keys()):
        if size_counters[size] != max_counter:
            del size_counters[size]
    
    if len(size_counters) != 1:
        raise ValueError("Ambiguous what image size is the most popular.")
    
    # The most popular image size
    size = list(size_counters.keys())[0]

    # pick the image with the most popular size
    image_with_size_list = [image for image in images if image.shape == size]
    if len(image_with_size_list) == 0:
        raise ValueError("No image with the most popular size.")

    image0 = image_with_size_list[0]
    height, width = image0.shape
    vote = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            histogram = Histogram.empty()
            for image in image_with_size_list:
                pixel = image[y, x]
                histogram.increment(pixel)
            color = histogram.most_popular_color()
            if color is None:
                color = 255
            vote[y, x] = color
    return vote
