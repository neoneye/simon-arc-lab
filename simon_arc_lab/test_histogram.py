import unittest
import numpy as np
from .image_util import *
from .histogram import *

class TestHistogram(unittest.TestCase):
    def test_sorted_color_count_list_unambiguous(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [5, 9],
            [6, 9],
            [6, 9]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.sorted_color_count_list()
        expected = [(9, 3), (6, 2), (5, 1)]
        self.assertTrue(actual, expected)

    def test_sorted_histogram_of_image_tie(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [9, 9],
            [5, 5],
            [7, 7]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.sorted_color_count_list()
        expected = [(5, 2), (7, 2), (9, 2)]
        self.assertTrue(actual, expected)

    def test_pretty_histogram_of_image_unambiguous(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [5, 9],
            [6, 9],
            [6, 9]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.pretty()
        expected = '9:3,6:2,5:1'
        self.assertTrue(actual, expected)

    def test_pretty_histogram_of_image_tie(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [9, 9],
            [5, 5],
            [7, 7]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.pretty()
        expected = '5:2,7:2,9:2'
        self.assertTrue(actual, expected)

    def test_pretty_histogram_of_image_tie2(self):
        image = np.zeros((3, 4), dtype=np.uint8)
        image[0:3, 0:4] = [
            [9, 9, 5, 5],
            [5, 5, 7, 7],
            [7, 7, 9, 9]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.pretty()
        expected = '5:4,7:4,9:4'
        self.assertTrue(actual, expected)

if __name__ == '__main__':
    unittest.main()
