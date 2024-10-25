import unittest
import numpy as np
from .image_util import *
from .histogram import *

class TestHistogram(unittest.TestCase):
    def test_init_with_purge(self):
        histogram = Histogram({8: 0, 9: -1, 7: 7, 5: 5})
        actual = histogram.pretty()
        expected = '7:7,5:5'
        self.assertEqual(actual, expected)

    def test_pretty_empty0(self):
        actual = Histogram.empty().pretty()
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_pretty_empty1(self):
        actual = Histogram({5:0}).pretty()
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_pretty_nonempty0(self):
        actual = Histogram({5:1}).pretty()
        expected = '5:1'
        self.assertEqual(actual, expected)

    def test_pretty_nonempty1(self):
        actual = Histogram({5:1,2:8}).pretty()
        expected = '2:8,5:1'
        self.assertEqual(actual, expected)

    def test_create_random0(self):
        actual = Histogram.create_random(0, 1, 1, 55, 55).pretty()
        expected = '7:55'
        self.assertEqual(actual, expected)

    def test_create_random1(self):
        actual = Histogram.create_random(1, 2, 2, 55, 55).pretty()
        expected = '6:55,8:55'
        self.assertEqual(actual, expected)

    def test_create_random2(self):
        actual = Histogram.create_random(1, 3, 3, 55, 55).pretty()
        expected = '6:55,8:55,9:55'
        self.assertEqual(actual, expected)

    def test_create_random3(self):
        actual = Histogram.create_random(1, 9, 9, 20, 30).pretty()
        expected = '4:29,7:29,9:29,0:27,1:27,5:25,3:23,6:23,8:23'
        self.assertEqual(actual, expected)

    def test_create_random4(self):
        for _ in range(100):
            histogram = Histogram.create_random(1, 1, 9, 20, 55)
            self.assertTrue(histogram.number_of_unique_colors() >= 1)
            self.assertTrue(histogram.number_of_unique_colors() <= 10)

    def test_create_with_image_list(self):
        image0 = np.array([
            [5, 5, 5], 
            [5, 5, 9]], dtype=np.uint8)
        image1 = np.array([
            [7, 8, 7, 8], 
            [7, 8, 7, 8],
            [7, 8, 7, 9]], dtype=np.uint8)
        histogram = Histogram.create_with_image_list([image0, image1])
        actual = histogram.pretty()
        expected = '7:6,5:5,8:5,9:2'
        self.assertEqual(actual, expected)

    def test_sorted_color_count_list_unambiguous(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [5, 9],
            [6, 9],
            [6, 9]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.sorted_color_count_list()
        expected = [(9, 3), (6, 2), (5, 1)]
        self.assertEqual(actual, expected)

    def test_sorted_histogram_of_image_tie(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [9, 9],
            [5, 5],
            [7, 7]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.sorted_color_count_list()
        expected = [(5, 2), (7, 2), (9, 2)]
        self.assertEqual(actual, expected)

    def test_sorted_count_list(self):
        image = np.array([
            [9, 8, 7, 7, 6, 6, 6, 5, 5, 5, 5]], dtype=np.uint8)
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.sorted_count_list()
        expected = [4, 3, 2, 1, 1]
        self.assertEqual(actual, expected)

    def test_pretty_histogram_of_image_unambiguous(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [5, 9],
            [6, 9],
            [6, 9]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.pretty()
        expected = '9:3,6:2,5:1'
        self.assertEqual(actual, expected)

    def test_pretty_histogram_of_image_tie(self):
        image = np.zeros((3, 2), dtype=np.uint8)
        image[0:3, 0:2] = [
            [9, 9],
            [5, 5],
            [7, 7]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.pretty()
        expected = '5:2,7:2,9:2'
        self.assertEqual(actual, expected)

    def test_pretty_histogram_of_image_tie2(self):
        image = np.zeros((3, 4), dtype=np.uint8)
        image[0:3, 0:4] = [
            [9, 9, 5, 5],
            [5, 5, 7, 7],
            [7, 7, 9, 9]]
        
        histogram = Histogram.create_with_image(image)
        actual = histogram.pretty()
        expected = '5:4,7:4,9:4'
        self.assertEqual(actual, expected)

    def test_increment_existing(self):
        histogram = Histogram({0: 8, 1: 2, 5: 1})
        histogram.increment(0)
        self.assertEqual(histogram.pretty(), '0:9,1:2,5:1')

    def test_increment_insert_a(self):
        histogram = Histogram({0: 8, 1: 2, 5: 1})
        histogram.increment(4)
        self.assertEqual(histogram.pretty(), '0:8,1:2,4:1,5:1')

    def test_increment_insert_b(self):
        histogram = Histogram({0: 8, 1: 2, 5: 1})
        histogram.increment(6)
        self.assertEqual(histogram.pretty(), '0:8,1:2,5:1,6:1')

    def test_add(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.add(histogram1).pretty()
        expected = '0:10,1:10,8:2,9:1'
        self.assertEqual(actual, expected)

    def test_subtract_and_discard0(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.subtract_and_discard(histogram1).pretty()
        expected = '0:6,9:1'
        self.assertEqual(actual, expected)

    def test_subtract_and_discard1(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram.empty()
        actual = histogram0.subtract_and_discard(histogram1).pretty()
        expected = '0:8,1:2,9:1'
        self.assertEqual(actual, expected)

    def test_subtract_and_discard2(self):
        histogram0 = Histogram.empty()
        histogram1 = Histogram({0: 8, 1: 2, 9: 1})
        actual = histogram0.subtract_and_discard(histogram1).pretty()
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_max(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.max(histogram1).pretty()
        expected = '0:8,1:8,8:2,9:1'
        self.assertEqual(actual, expected)

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
        actual = histogram0.min(histogram1).pretty()
        expected = '0:2,1:2'
        self.assertEqual(actual, expected)

    def test_number_of_unique_colors0(self):
        actual = Histogram.empty().number_of_unique_colors()
        self.assertEqual(actual, 0)

    def test_number_of_unique_colors1(self):
        actual = Histogram({9:1}).number_of_unique_colors()
        self.assertEqual(actual, 1)

    def test_number_of_unique_colors2(self):
        dict = {}
        for i in range(10):
            dict[i] = 1
        actual = Histogram(dict).number_of_unique_colors()
        self.assertEqual(actual, 10)

    def test_unique_colors0(self):
        actual = Histogram.empty().unique_colors()
        expected = []
        self.assertEqual(actual, expected)

    def test_unique_colors1(self):
        actual = Histogram({5:1,2:8}).unique_colors()
        expected = [2, 5]
        self.assertEqual(actual, expected)

    def test_unique_colors_set0(self):
        actual = Histogram.empty().unique_colors_set()
        expected = set()
        self.assertEqual(actual, expected)

    def test_unique_colors_set1(self):
        actual = Histogram({5:1,2:8,9:0}).unique_colors_set()
        expected = set([5, 2])
        self.assertEqual(actual, expected)

    def test_unique_colors_pretty0(self):
        actual = Histogram.empty().unique_colors_pretty()
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_unique_colors_pretty1(self):
        actual = Histogram({9:10}).unique_colors_pretty()
        expected = '9'
        self.assertEqual(actual, expected)

    def test_unique_colors_pretty2(self):
        actual = Histogram({5:1,2:8}).unique_colors_pretty()
        expected = '2,5'
        self.assertEqual(actual, expected)

    def test_unique_colors_pretty3(self):
        actual = Histogram({9:1,2:8,5:5}).unique_colors_pretty()
        expected = '2,5,9'
        self.assertEqual(actual, expected)

    def test_color_intersection_pretty0(self):
        histogram0 = Histogram.empty()
        histogram1 = Histogram.empty()
        actual = histogram0.color_intersection_pretty(histogram1)
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_color_intersection_pretty1(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram.empty()
        actual = histogram0.color_intersection_pretty(histogram1)
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_color_intersection_pretty2(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.color_intersection_pretty(histogram1)
        expected = '0,1'
        self.assertEqual(actual, expected)

    def test_remove_other_colors_no_overlap(self):
        histogram0 = Histogram({0: 8, 1: 2, 3: 1})
        histogram1 = Histogram({4: 2, 5: 8, 6: 2})
        actual = histogram0.remove_other_colors(histogram1).pretty()
        expected = '0:8,1:2,3:1'
        self.assertEqual(actual, expected)

    def test_remove_other_colors_some_overlap(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2})
        actual = histogram0.remove_other_colors(histogram1).pretty()
        expected = '9:1'
        self.assertEqual(actual, expected)

    def test_remove_other_colors_full_overlap(self):
        histogram0 = Histogram({0: 8, 1: 2, 9: 1})
        histogram1 = Histogram({0: 2, 1: 8, 8: 2, 9: 10})
        actual = histogram0.remove_other_colors(histogram1).pretty()
        expected = 'empty'
        self.assertEqual(actual, expected)

    def test_sum_of_counters0(self):
        actual = Histogram.empty().sum_of_counters()
        self.assertEqual(actual, 0)

    def test_sum_of_counters1(self):
        actual = Histogram({9: 1}).sum_of_counters()
        self.assertEqual(actual, 1)

    def test_sum_of_counters2(self):
        actual = Histogram({9: 1, 8: 2, 7: 3}).sum_of_counters()
        self.assertEqual(actual, 6)

    def test_most_popular_color_unambiguous(self):
        actual = Histogram({0: 5, 6: 3, 7: 8}).most_popular_color()
        self.assertEqual(actual, 7)

    def test_most_popular_color_tie_a(self):
        actual = Histogram({0: 5, 6: 8, 7: 8}).most_popular_color()
        self.assertEqual(actual, None)

    def test_most_popular_color_tie_b(self):
        actual = Histogram({0: 5, 6: 8, 7: 8, 8: 9, 9: 9}).most_popular_color()
        self.assertEqual(actual, None)

    def test_most_popular_color_tie_c(self):
        actual = Histogram({0: 1, 1: 1, 2: 2, 3: 2, 6: 8, 7: 8, 8: 9, 9: 9}).most_popular_color()
        self.assertEqual(actual, None)

    def test_most_popular_color_empty(self):
        actual = Histogram({}).most_popular_color()
        self.assertEqual(actual, None)

    def test_most_popular_color_zero(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: 0}
        actual = histogram.most_popular_color()
        self.assertEqual(actual, None)

    def test_most_popular_color_negative(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: -1}
        actual = histogram.most_popular_color()
        self.assertEqual(actual, None)

    def test_most_popular_color_list_unambiguous(self):
        actual = Histogram({0: 5, 6: 3, 7: 8}).most_popular_color_list()
        self.assertEqual(actual, [7])

    def test_most_popular_color_list_tie_a(self):
        actual = Histogram({0: 5, 6: 8, 7: 8}).most_popular_color_list()
        self.assertEqual(actual, [6, 7])

    def test_most_popular_color_list_tie_b(self):
        actual = Histogram({0: 5, 6: 8, 7: 8, 8: 9, 9: 9}).most_popular_color_list()
        self.assertEqual(actual, [8, 9])

    def test_most_popular_color_list_tie_c(self):
        actual = Histogram({0: 1, 1: 1, 2: 9, 3: 2, 6: 8, 7: 8, 8: 9, 9: 9}).most_popular_color_list()
        self.assertEqual(actual, [2, 8, 9])

    def test_most_popular_color_list_empty(self):
        actual = Histogram({}).most_popular_color_list()
        self.assertEqual(actual, [])

    def test_most_popular_color_list_zero(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: 0}
        actual = histogram.most_popular_color_list()
        self.assertEqual(actual, [])

    def test_most_popular_color_list_negative(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: -1}
        actual = histogram.most_popular_color_list()
        self.assertEqual(actual, [])

    def test_least_popular_color_unambiguous(self):
        actual = Histogram({0: 5, 6: 1, 7: 8}).least_popular_color()
        self.assertEqual(actual, 6)

    def test_least_popular_color_tie_a(self):
        actual = Histogram({0: 9, 6: 8, 7: 8}).least_popular_color()
        self.assertEqual(actual, None)

    def test_least_popular_color_tie_b(self):
        actual = Histogram({0: 5, 6: 8, 7: 5, 8: 9, 9: 9}).least_popular_color()
        self.assertEqual(actual, None)

    def test_least_popular_color_tie_c(self):
        actual = Histogram({0: 1, 1: 1, 2: 2, 3: 2, 6: 8, 7: 8, 8: 9, 9: 9}).least_popular_color()
        self.assertEqual(actual, None)

    def test_least_popular_color_empty(self):
        actual = Histogram({}).least_popular_color()
        self.assertEqual(actual, None)

    def test_least_popular_color_zero(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: 0}
        actual = histogram.least_popular_color()
        self.assertEqual(actual, None)

    def test_least_popular_color_negative(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: -1}
        actual = histogram.least_popular_color()
        self.assertEqual(actual, None)

    def test_least_popular_color_list_unambiguous(self):
        actual = Histogram({0: 5, 6: 1, 7: 8}).least_popular_color_list()
        self.assertEqual(actual, [6])

    def test_least_popular_color_list_tie_a(self):
        actual = Histogram({0: 9, 6: 8, 7: 8}).least_popular_color_list()
        self.assertEqual(actual, [6, 7])

    def test_least_popular_color_list_tie_b(self):
        actual = Histogram({0: 5, 6: 8, 7: 5, 8: 9, 9: 9}).least_popular_color_list()
        self.assertEqual(actual, [0, 7])

    def test_least_popular_color_list_tie_c(self):
        actual = Histogram({0: 1, 1: 1, 2: 2, 3: 2, 6: 8, 7: 8, 8: 9, 9: 9}).least_popular_color_list()
        self.assertEqual(actual, [0, 1])

    def test_least_popular_color_list_empty(self):
        actual = Histogram({}).least_popular_color_list()
        self.assertEqual(actual, [])

    def test_least_popular_color_list_zero(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: 0}
        actual = histogram.least_popular_color_list()
        self.assertEqual(actual, [])

    def test_least_popular_color_list_negative(self):
        histogram = Histogram.empty()
        histogram.color_count = {5: -1}
        actual = histogram.least_popular_color_list()
        self.assertEqual(actual, [])

    def test_get_count_for_color(self):
        histogram = Histogram({0: 5, 6: 1, 7: 8})
        self.assertEqual(histogram.get_count_for_color(0), 5)
        self.assertEqual(histogram.get_count_for_color(6), 1)
        self.assertEqual(histogram.get_count_for_color(7), 8)
        self.assertEqual(histogram.get_count_for_color(9), 0)

    def test_available_colors(self):
        self.assertEqual(Histogram.empty().available_colors(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(Histogram({0: -5}).available_colors(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(Histogram({0: 5, 6: 1, 7: 8}).available_colors(), [1, 2, 3, 4, 5, 8, 9])
        self.assertEqual(Histogram({0: 1, 1: 7, 2: 0}).available_colors(), [2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(Histogram({0: 1, 1: 7, 2: 1}).available_colors(), [3, 4, 5, 6, 7, 8, 9])
        histogram = Histogram.empty()
        for color in range(10):
            histogram.increment(color)
        self.assertEqual(histogram.available_colors(), [])

    def test_first_available_color(self):
        self.assertEqual(Histogram.empty().first_available_color(), 0)
        self.assertEqual(Histogram({0: -5}).first_available_color(), 0)
        self.assertEqual(Histogram({0: 5, 6: 1, 7: 8}).first_available_color(), 1)
        self.assertEqual(Histogram({0: 1, 1: 7, 2: 0}).first_available_color(), 2)
        self.assertEqual(Histogram({0: 1, 1: 7, 2: 1}).first_available_color(), 3)
        histogram = Histogram.empty()
        for color in range(10):
            histogram.increment(color)
        self.assertEqual(histogram.first_available_color(), None)

    def test_remove_color_existing(self):
        histogram = Histogram({9:1,2:8,5:5,255:999})
        histogram.remove_color(255)
        actual = histogram.pretty()
        expected = '2:8,5:5,9:1'
        self.assertEqual(actual, expected)

    def test_remove_color_nonexisting(self):
        histogram = Histogram({9:1,2:8,5:5,255:999})
        histogram.remove_color(42)
        actual = histogram.pretty()
        expected = '255:999,2:8,5:5,9:1'
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
