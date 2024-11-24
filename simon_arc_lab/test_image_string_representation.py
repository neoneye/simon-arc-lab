import unittest
import numpy as np
from .image_string_representation import *

class TestImageStringRepresentation(unittest.TestCase):
    def test_image_to_string(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        actual = image_to_string(image)
        expected = "123\n456"
        self.assertEqual(actual, expected)

    def test_image_from_string(self):
        image = "123\n456"
        actual = image_from_string(image)
        expected = np.array([
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
