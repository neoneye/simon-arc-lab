import unittest
import numpy as np
from .image_from_json_array import image_from_json_array

class TestImageFromJSONArray(unittest.TestCase):
    def test_10000_different_widths(self):
        json_array = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]
        actual = image_from_json_array(json_array)
        expected = np.array([
            [1, 2, 3, 255], 
            [4, 5, 255, 255], 
            [6, 7, 8, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
