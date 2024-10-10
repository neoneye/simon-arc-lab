import unittest
import numpy as np
from .image_similarity import *

class TestImageSimilarity(unittest.TestCase):
    def test_10000_compute_jaccard_index(self):
        self.assertEqual(ImageSimilarity.compute_jaccard_index([]), 0)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([False]), 0)
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
        i = ImageSimilarity.create_with_images(image0, image0)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_unique_colors()
        # Assert
        self.assertEqual(actual, False)

    def test_32000_same_number_of_unique_colors_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [9, 8, 7],
            [9, 8, 7],
            [6, 5, 4]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_number_of_unique_colors()
        # Assert
        self.assertEqual(actual, True)

    def test_32001_same_number_of_unique_colors_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_number_of_unique_colors()
        # Assert
        self.assertEqual(actual, True)

    def test_32002_same_number_of_unique_colors_false(self):
        # Arrange
        image0 = np.array([
            [3, 4, 5],
            [6, 7, 8]], dtype=np.uint8)
        image1 = np.array([
            [3, 4, 5],
            [6, 7, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_number_of_unique_colors()
        # Assert
        self.assertEqual(actual, False)

    def test_33002_same_histogram_ignoring_scale_factor1(self):
        # Arrange
        image0 = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        image1 = np.array([
            [4, 3, 2, 1]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, True)

    def test_33001_same_histogram_ignoring_scale_factor2(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, True)

    def test_33002_same_histogram_ignoring_scale_factor3(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, True)

    def test_33003_same_histogram_ignoring_scale_false(self):
        # Arrange
        image0 = np.array([
            [1],
            [2],
            [3]], dtype=np.uint8)
        image1 = np.array([
            [1, 2]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_histogram_ignoring_scale()
        # Assert
        self.assertEqual(actual, False)

    def test_34000_same_histogram_counters_true(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [6, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 8, 8, 8, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_histogram_counters()
        # Assert
        self.assertEqual(actual, True)

    def test_34001_same_histogram_counters_false(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [6, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 8, 8, 8, 8, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_histogram_counters()
        # Assert
        self.assertEqual(actual, False)

    def test_35000_same_most_popular_color_list_unambiguous_true(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [5, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 5, 8, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_most_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_35001_same_most_popular_color_list_tie_true(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_most_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_35002_same_most_popular_color_list_false(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1],
            [1, 7, 1],
            [1, 7, 1],
            [1, 1, 1]], dtype=np.uint8)
        image1 = np.array([
            [3, 3, 3, 1]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_most_popular_color_list()
        # Assert
        self.assertEqual(actual, False)

    def test_36000_same_least_popular_color_list_unambiguous_true(self):
        # Arrange
        image0 = np.array([
            [5, 5],
            [5, 6],
            [6, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 5, 7, 1, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_least_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_36001_same_least_popular_color_list_tie_true(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_least_popular_color_list()
        # Assert
        self.assertEqual(actual, True)

    def test_36002_same_least_popular_color_list_false(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1],
            [1, 7, 1],
            [1, 7, 1],
            [1, 1, 1]], dtype=np.uint8)
        image1 = np.array([
            [7, 7, 7, 1, 1]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_least_popular_color_list()
        # Assert
        self.assertEqual(actual, False)

    def test_37000_agree_on_color_a_color_that_is_present(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.agree_on_color(7)
        # Assert
        self.assertEqual(actual, True)

    def test_37001_agree_on_color_a_color_that_isnt_present(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.agree_on_color(6)
        # Assert
        self.assertEqual(actual, True)

    def test_37002_same_color_is_present_false(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.agree_on_color(9)
        # Assert
        self.assertEqual(actual, False)

    def test_38000_agree_on_color_with_same_counter_true_due_to_same_counter_value(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 7, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.agree_on_color_with_same_counter(7)
        # Assert
        self.assertEqual(actual, True)

    def test_38001_agree_on_color_with_same_counter_true_due_to_counter_being_zero(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 7, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.agree_on_color_with_same_counter(8)
        # Assert
        self.assertEqual(actual, True)

    def test_38002_agree_on_color_with_same_counter_false_due_to_different_counters(self):
        # Arrange
        image0 = np.array([
            [1, 1],
            [1, 1],
            [5, 7],
            [5, 7],
            [5, 7]], dtype=np.uint8)
        image1 = np.array([
            [5, 5, 7, 9, 9, 9, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.agree_on_color_with_same_counter(7)
        # Assert
        self.assertEqual(actual, False)

    def test_39000_unique_colors_is_a_subset_true(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 2, 3, 2, 3]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.unique_colors_is_a_subset()
        # Assert
        self.assertEqual(actual, True)

    def test_39001_unique_colors_is_a_subset_true(self):
        # Arrange
        image0 = np.array([
            [3, 2, 3, 2, 3]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.unique_colors_is_a_subset()
        # Assert
        self.assertEqual(actual, True)

    def test_39002_unique_colors_is_a_subset_false(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [5, 6, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.unique_colors_is_a_subset()
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
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
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_direction_topbottom()
        # Assert
        self.assertEqual(actual, False)

    def test_40300_same_bigrams_direction_topleftbottomright_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [2, 1, 4, 4],
            [3, 2, 5, 4],
            [3, 3, 6, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_direction_topleftbottomright()
        # Assert
        self.assertEqual(actual, True)

    def test_40301_same_bigrams_direction_topleftbottomright_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 6]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_direction_topleftbottomright()
        # Assert
        self.assertEqual(actual, False)

    def test_40400_same_bigrams_direction_topleftbottomright_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 6]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_direction_toprightbottomleft()
        # Assert
        self.assertEqual(actual, True)

    def test_40401_same_bigrams_direction_topleftbottomright_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [2, 1, 4, 4],
            [3, 2, 5, 4],
            [3, 3, 6, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_direction_toprightbottomleft()
        # Assert
        self.assertEqual(actual, False)

    def test_40500_same_bigrams_subset_crop(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 1, 4, 9],
            [9, 2, 5, 9],
            [9, 3, 6, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_subset()
        # Assert
        self.assertEqual(actual, True)

    def test_40501_same_bigrams_subset_flip(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 4, 1, 9],
            [9, 5, 2, 9],
            [9, 6, 3, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_subset()
        # Assert
        self.assertEqual(actual, True)

    def test_40502_same_bigrams_subset_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 5, 5, 9],
            [9, 5, 5, 9],
            [9, 5, 5, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bigrams_subset()
        # Assert
        self.assertEqual(actual, False)

    def test_50000_same_bounding_box_size_of_color_a_color_that_is_present(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [9, 9, 5, 9, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 5, 5, 9],
            [9, 9, 9, 9],
            [9, 5, 5, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bounding_box_size_of_color(5)
        # Assert
        self.assertEqual(actual, True)

    def test_50001_same_bounding_box_size_of_color_a_color_that_is_not_present(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [9, 9, 5, 9, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 5, 5, 9],
            [9, 9, 9, 9],
            [9, 5, 5, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bounding_box_size_of_color(0)
        # Assert
        self.assertEqual(actual, True)

    def test_50002_same_bounding_box_size_of_color_false(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [9, 9, 5, 9, 9, 9],
            [9, 5, 5, 9, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 5, 5, 9],
            [9, 5, 9, 9],
            [9, 5, 5, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bounding_box_size_of_color(5)
        # Assert
        self.assertEqual(actual, False)

    def test_50003_same_bounding_box_size_of_color_a_color_only_present_in_one_of_the_images_false(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [9, 9, 5, 9, 9, 9],
            [9, 5, 5, 9, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9],
            [9, 3, 3, 9],
            [9, 3, 3, 9],
            [9, 3, 3, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_bounding_box_size_of_color(5)
        # Assert
        self.assertEqual(actual, False)

    def test_60000_same_shape2x2_true(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [1, 5, 5, 1, 5, 5],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [1, 5, 5, 1, 5, 5, 1, 1, 5],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape2x2()
        # Assert
        self.assertEqual(actual, True)

    def test_60001_same_shape2x2_true(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [1, 5, 5, 1, 5, 5],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 9, 9],
            [7, 3, 3, 1, 5, 5],
            [7, 3, 3, 1, 5, 5],
            [7, 3, 3, 1, 5, 5],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape2x2()
        # Assert
        self.assertEqual(actual, True)

    def test_60002_same_shape2x2_false(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [1, 5, 5, 1, 5, 5],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 7, 7],
            [1, 5, 5, 1, 7, 1],
            [1, 7, 7, 7, 1, 7],
            [1, 5, 5, 1, 7, 1],
            [9, 9, 9, 9, 7, 7]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape2x2()
        # Assert
        self.assertEqual(actual, False)

    def test_70000_same_shape3x3opposite_true(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 9, 9],
            [9, 5, 9, 5, 9, 9],
            [9, 5, 5, 5, 9, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 5, 5, 5, 9],
            [9, 9, 5, 9, 5, 9],
            [9, 9, 5, 5, 5, 9],
            [9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape3x3opposite()
        # Assert
        self.assertEqual(actual, True)

    def test_70001_same_shape3x3opposite_false(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape3x3opposite()
        # Assert
        self.assertEqual(actual, False)

    def test_80000_same_shape3x3center_true(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5],
            [9, 5, 9, 5, 9, 5],
            [9, 5, 5, 5, 9, 5],
            [9, 9, 9, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5],
            [9, 5, 9, 5, 7, 5],
            [9, 5, 5, 5, 7, 5],
            [9, 9, 9, 5, 5, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape3x3center()
        # Assert
        self.assertEqual(actual, True)

    def test_80001_same_shape3x3center_false(self):
        # Arrange
        image0 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5],
            [9, 5, 9, 5, 9, 5],
            [9, 5, 5, 5, 9, 5],
            [9, 9, 9, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5],
            [9, 5, 9, 5, 5, 5],
            [9, 5, 5, 5, 5, 5],
            [9, 9, 9, 5, 5, 5]], dtype=np.uint8)
        i = ImageSimilarity.create_with_images(image0, image1)
        # Act
        actual = i.same_shape3x3center()
        # Assert
        self.assertEqual(actual, False)

    def test_90000_format_feature_list(self):
        # Arrange
        feature_list = [
            Feature(FeatureType.SAME_ORIENTATION),
            Feature(FeatureType.AGREE_ON_COLOR, 5),
            Feature(FeatureType.SAME_BIGRAMS_SUBSET),
        ]
        # Act
        actual = Feature.format_feature_list(feature_list)
        # Assert
        expected = 'agree_on_color(5),same_bigrams_subset,same_shape_orientation'
        self.assertEqual(actual, expected)
