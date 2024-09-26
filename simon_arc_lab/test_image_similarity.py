import unittest
import numpy as np
from .image_similarity import *

class TestImageSimilarity(unittest.TestCase):
    def test_10000_compute_jaccard_index(self):
        self.assertEqual(ImageSimilarity.compute_jaccard_index([False, False]), 0)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([False, True]), 50)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([True, False]), 50)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([True, True]), 100)

    def test_20000_same_image_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image0)
        # Act
        actual = i.same_image()
        # Assert
        self.assertEqual(actual, True)

    def test_20001_same_image_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 3],
            [3, 3],
            [3, 3]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_image()
        # Assert
        self.assertEqual(actual, False)

    def test_21000_same_shape_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 6],
            [2, 5],
            [1, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape()
        # Assert
        self.assertEqual(actual, True)

    def test_21001_same_shape_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape()
        # Assert
        self.assertEqual(actual, False)

    def test_22000_same_shape_allow_for_rotation_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 6],
            [2, 5],
            [1, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_allow_for_rotation()
        # Assert
        self.assertEqual(actual, True)

    def test_22001_same_shape_allow_for_rotation_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_allow_for_rotation()
        # Assert
        self.assertEqual(actual, True)

    def test_22002_same_shape_allow_for_rotation_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_allow_for_rotation()
        # Assert
        self.assertEqual(actual, False)

    def test_23000_same_shape_width_true(self):
        # Arrange
        image0 = np.array([
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_width()
        # Assert
        self.assertEqual(actual, True)

    def test_23001_same_shape_width_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_width()
        # Assert
        self.assertEqual(actual, False)

    def test_24000_same_shape_height_true(self):
        # Arrange
        image0 = np.array([
            [5, 5, 5],
            [5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_height()
        # Assert
        self.assertEqual(actual, True)

    def test_24001_same_shape_height_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_height()
        # Assert
        self.assertEqual(actual, False)

    def test_25000_same_shape_orientation_landscape(self):
        # Arrange
        image0 = np.array([
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_orientation()
        # Assert
        self.assertEqual(actual, True)

    def test_25001_same_shape_orientation_portrait(self):
        # Arrange
        image0 = np.array([
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_orientation()
        # Assert
        self.assertEqual(actual, True)

    def test_25002_same_shape_orientation_square(self):
        # Arrange
        image0 = np.array([
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [1, 3],
            [2, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_orientation()
        # Assert
        self.assertEqual(actual, True)

    def test_25003_same_shape_orientation_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_shape_orientation()
        # Assert
        self.assertEqual(actual, False)

    def test_30000_same_histogram_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 6],
            [2, 5],
            [1, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram()
        # Assert
        self.assertEqual(actual, True)

    def test_30001_same_histogram_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 3],
            [3, 3],
            [3, 3]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram()
        # Assert
        self.assertEqual(actual, False)

    def test_31000_same_unique_colors_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_unique_colors()
        # Assert
        self.assertEqual(actual, True)

    def test_31001_same_unique_colors_false(self):
        # Arrange
        image0 = np.array([
            [3, 4, 5],
            [6, 7, 8]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_unique_colors()
        # Assert
        self.assertEqual(actual, False)

    def test_32002_same_histogram_ignoring_scale_factor1(self):
        # Arrange
        image0 = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        image1 = np.array([
            [4, 3, 2, 1]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, True)

    def test_32001_same_histogram_ignoring_scale_factor2(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, True)

    def test_32002_same_histogram_ignoring_scale_factor3(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, True)

    def test_32003_same_histogram_ignoring_scale_false(self):
        # Arrange
        image0 = np.array([
            [1],
            [2],
            [3]], dtype=np.uint8)
        image1 = np.array([
            [1, 2]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, False)

    def test_33000_same_histogram_counters_true(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [6, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 8, 8, 8, 9]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram_counters()
        # Assert
        self.assertEqual(actual, True)

    def test_33001_same_histogram_counters_false(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [6, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 8, 8, 8, 8, 9]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram_counters()
        # Assert
        self.assertEqual(actual, False)

    def test_34000_same_most_popular_color_list_unambiguous_true(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [5, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 5, 8, 5]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_most_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_34001_same_most_popular_color_list_tie_true(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_most_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_34002_same_most_popular_color_list_false(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1],
            [1, 7, 1],
            [1, 7, 1],
            [1, 1, 1]], dtype=np.uint8)
        image1 = np.array([
            [3, 3, 3, 1]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_most_popular_color_list()
        # Assert
        self.assertEqual(actual, False)

    def test_35000_same_least_popular_color_list_unambiguous_true(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [5, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 5, 7, 1, 5]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_least_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_35001_same_least_popular_color_list_tie_true(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_least_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_35002_same_least_popular_color_list_false(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1],
            [1, 7, 1],
            [1, 7, 1],
            [1, 1, 1]], dtype=np.uint8)
        image1 = np.array([
            [7, 7, 7, 1, 1]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_least_popular_color_list()
        # Assert
        self.assertEqual(actual, False)

    def test_36000_agree_on_color_a_color_that_is_present(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.agree_on_color(7)
        # Assert
        self.assertEqual(actual, True)

    def test_36001_agree_on_color_a_color_that_isnt_present(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.agree_on_color(6)
        # Assert
        self.assertEqual(actual, True)

    def test_36002_same_color_is_present_false(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.agree_on_color(9)
        # Assert
        self.assertEqual(actual, False)

    def test_40000_same_bigrams_direction_all_rotate(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_all()
        # Assert
        self.assertEqual(actual, True)

    def test_40001_same_bigrams_direction_all_flip(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 2, 1],
            [6, 5, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_all()
        # Assert
        self.assertEqual(actual, True)

    def test_40002_same_bigrams_direction_all_symmetry(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 2, 1, 2, 3],
            [6, 5, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_all()
        # Assert
        self.assertEqual(actual, True)

    def test_40003_same_bigrams_direction_all_false(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [6, 5, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_all()
        # Assert
        self.assertEqual(actual, False)

    def test_40004_same_bigrams_direction_all_false(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 5],
            [7, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_all()
        # Assert
        self.assertEqual(actual, False)

    def test_40100_same_bigrams_direction_leftright_true(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 1, 2, 3],
            [4, 5, 5, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_leftright()
        # Assert
        self.assertEqual(actual, True)

    def test_40101_same_bigrams_direction_leftright_false(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [2, 1, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_leftright()
        # Assert
        self.assertEqual(actual, False)

    def test_40200_same_bigrams_direction_topbottom_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 4],
            [2, 5],
            [1, 5],
            [2, 6],
            [3, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_topbottom()
        # Assert
        self.assertEqual(actual, True)

    def test_40201_same_bigrams_direction_topbottom_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 4],
            [3, 5],
            [2, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_bigrams_direction_topbottom()
        # Assert
        self.assertEqual(actual, False)

