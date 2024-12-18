import unittest
import numpy as np
from .image_to_python import *

def process(image: np.array, max_depth: int, verbose: bool=False) -> str:
    config = ImageToPythonConfig()
    config.max_depth=max_depth
    config.verbose=verbose
    image_to_python = ImageToPython(image, config)
    image_to_python.build()
    return image_to_python.python_code

class TestImageToPython(unittest.TestCase):
    def test_10000_solid_color(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        process(image, 1)

    def test_11000_direction_topbottom(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        process(image, 2)

    def test_12000_direction_leftright(self):
        image = np.array([
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1]], dtype=np.uint8)
        process(image, 3)

    def test_13000_direction_leftright_and_direction_topbottom(self):
        image = np.array([
            [5, 5, 3, 1],
            [7, 7, 7, 7],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1]], dtype=np.uint8)
        process(image, 4)

    def test_20000_small_image_with_one_pixel_in_the_center(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        image[1, 1] = 42
        process(image, 4)

    def test_20001_big_image_with_one_pixel_in_the_center(self):
        image = np.zeros((99, 51), dtype=np.uint8)
        image[50, 25] = 1
        process(image, 8)

    def test_20002_big_image_with_2x2_pixels_in_the_center(self):
        image = np.zeros((99, 51), dtype=np.uint8)
        image[50:52, 25:27] = 1
        process(image, 10)
