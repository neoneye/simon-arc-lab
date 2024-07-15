import unittest
import numpy as np
from .image_util import *
from .histogram import *

class TestHistogram(unittest.TestCase):
    def test_init_with_zero_purge_zeros(self):
        histogram = Histogram({8: 0, 7: 7, 5: 5})
        actual = histogram.pretty()
        expected = '5:5,7:7'
        self.assertTrue(actual, expected)

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

    def test_add(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.add(histogram1).pretty()
        expected = '0:10,1:10,8:2,9:1'
        self.assertTrue(actual, expected)

    def test_max(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.add(histogram1).pretty()
        expected = '0:8,1:8,8:2,9:1'
        self.assertTrue(actual, expected)

    def test_color_intersection_set0(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.color_intersection_set(histogram1)
        expected = set([0, 1])
        self.assertEqual(actual, expected)

    def test_color_intersection_set1(self):
        histogram0 = Histogram({1: 8, 2: 2})
        histogram1 = Histogram({3: 2, 4: 8})
        actual = histogram0.color_intersection_set(histogram1)
        expected = set([])
        self.assertEqual(actual, expected)

    def test_color_intersection_set2(self):
        histogram0 = Histogram.empty()
        histogram1 = Histogram.empty()
        actual = histogram0.color_intersection_set(histogram1)
        expected = set()
        self.assertEqual(actual, expected)

    def test_min(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.add(histogram1).pretty()
        expected = '0:2,1:2'
        self.assertTrue(actual, expected)

if __name__ == '__main__':
    unittest.main()
