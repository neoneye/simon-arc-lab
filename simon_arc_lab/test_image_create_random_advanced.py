import unittest
import numpy as np
from .image_util import *
from .image_create_random_advanced import *

class TestImageCreateRandomAdvanced(unittest.TestCase):
    def test_size_exact_1(self):
        for i in range(100):
            seed = i
            min_width = 5
            max_width = 5
            min_height = 1
            max_height = 3
            actual = image_create_random_advanced(seed, min_width, max_width, min_height, max_height)
            self.assertTrue(actual.shape[0] >= min_height)
            self.assertTrue(actual.shape[0] <= max_height)
            self.assertTrue(actual.shape[1] >= min_width)
            self.assertTrue(actual.shape[1] <= max_width)

    def test_size_exact_2(self):
        for i in range(100):
            seed = i + 1000
            min_width = 1
            max_width = 3
            min_height = 5
            max_height = 5
            actual = image_create_random_advanced(seed, min_width, max_width, min_height, max_height)
            self.assertTrue(actual.shape[0] >= min_height)
            self.assertTrue(actual.shape[0] <= max_height)
            self.assertTrue(actual.shape[1] >= min_width)
            self.assertTrue(actual.shape[1] <= max_width)

    def test_size_range_1(self):
        for i in range(100):
            seed = i + 2000
            min_width = 7
            max_width = 9
            min_height = 6
            max_height = 8
            actual = image_create_random_advanced(seed, min_width, max_width, min_height, max_height)
            self.assertTrue(actual.shape[0] >= min_height)
            self.assertTrue(actual.shape[0] <= max_height)
            self.assertTrue(actual.shape[1] >= min_width)
            self.assertTrue(actual.shape[1] <= max_width)

if __name__ == '__main__':
    unittest.main()
