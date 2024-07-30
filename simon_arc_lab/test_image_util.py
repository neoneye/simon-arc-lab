import unittest
import numpy as np
from .image_util import *

class TestImageUtil(unittest.TestCase):
    def test_image_create(self):
        actual = image_create(2, 3, 4)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [4, 4],
            [4, 4],
            [4, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_two_colors_50(self):
        actual = image_create_random_with_two_colors(2, 3, 4, 5, 0.5, 0)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [4, 5],
            [5, 4],
            [5, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_two_colors_25(self):
        actual = image_create_random_with_two_colors(2, 4, 8, 9, 0.25, 0)

        expected = np.zeros((4, 2), dtype=np.uint8)
        expected[0:4, 0:2] = [
            [8, 9],
            [8, 8],
            [9, 8],
            [8, 8]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_three_colors_1_1_1(self):
        actual = image_create_random_with_three_colors(2, 3, 4, 5, 6, 1, 1, 1, 0)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [6, 6],
            [5, 4],
            [5, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_three_colors_3_2_1(self):
        actual = image_create_random_with_three_colors(2, 3, 4, 5, 6, 3, 2, 1, 0)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [4, 6],
            [5, 4],
            [5, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_four_colors_1_1_1_1(self):
        actual = image_create_random_with_four_colors(2, 4, 4, 5, 6, 7, 1, 1, 1, 1, 0)

        expected = np.zeros((4, 2), dtype=np.uint8)
        expected[0:4, 0:2] = [
            [7, 5],
            [6, 7],
            [5, 6],
            [4, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_four_colors_3_3_1_1(self):
        actual = image_create_random_with_four_colors(2, 4, 4, 5, 6, 7, 3, 3, 1, 1, 0)

        expected = np.zeros((4, 2), dtype=np.uint8)
        expected[0:4, 0:2] = [
            [7, 5],
            [6, 4],
            [5, 5],
            [4, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_rotate_cw(self):
        input = np.zeros((3, 2), dtype=np.uint8)
        input[0:3, 0:2] = [
            [3, 6],
            [2, 5],
            [1, 4]]

        expected = np.zeros((2, 3), dtype=np.uint8)
        expected[0:2, 0:3] = [
            [1, 2, 3],
            [4, 5, 6]]

        actual = image_rotate_cw(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_rotate_ccw(self):
        input = np.zeros((3, 2), dtype=np.uint8)
        input[0:3, 0:2] = [
            [4, 1],
            [5, 2],
            [6, 3]]

        expected = np.zeros((2, 3), dtype=np.uint8)
        expected[0:2, 0:3] = [
            [1, 2, 3],
            [4, 5, 6]]

        actual = image_rotate_ccw(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_rotate_180(self):
        input = np.zeros((3, 2), dtype=np.uint8)
        input[0:3, 0:2] = [
            [4, 1],
            [5, 2],
            [6, 3]]

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [3, 6],
            [2, 5],
            [1, 4]]

        actual = image_rotate_180(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_bresenham_line_diagonal(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        actual = bresenham_line(image, 0, 0, 2, 2, 1)

        expected = np.zeros((3, 3), dtype=np.uint8)
        expected[0, 0] = 1
        expected[1, 1] = 1
        expected[2, 2] = 1

        self.assertTrue(np.array_equal(actual, expected))

    def test_bresenham_line_horizontal(self):
        image = np.zeros((3, 4), dtype=np.uint8)
        actual = bresenham_line(image, 0, 0, 3, 0, 1)

        expected = np.zeros((3, 4), dtype=np.uint8)
        expected[0, 0] = 1
        expected[0, 1] = 1
        expected[0, 2] = 1
        expected[0, 3] = 1

        self.assertTrue(np.array_equal(actual, expected))

    def test_bresenham_line_horizontal(self):
        image = np.zeros((3, 4), dtype=np.uint8)
        actual = bresenham_line(image, 1, 0, 1, 2, 1)

        expected = np.zeros((3, 4), dtype=np.uint8)
        expected[0, 1] = 1
        expected[1, 1] = 1
        expected[2, 1] = 1

        self.assertTrue(np.array_equal(actual, expected))

    def test_count_same_color_as_center_with_one_neighbor_nowrap_1(self):
        image = np.array([[1, 5, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        actual = count_same_color_as_center_with_one_neighbor_nowrap(image, 0, -1)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_count_same_color_as_center_with_one_neighbor_nowrap_2(self):
        image = np.array([[5, 2, 3], [4, 5, 6], [7, 8, 5]], dtype=np.uint8)
        actual = count_same_color_as_center_with_one_neighbor_nowrap(image, -1, -1)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_count_neighbors_with_same_color_nowrap_1(self):
        image = np.array([[9, 7, 9], [7, 7, 7], [9, 7, 9]], dtype=np.uint8)
        actual = count_neighbors_with_same_color_nowrap(image)
        expected = np.array([[0, 3, 0], [3, 4, 3], [0, 3, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_count_neighbors_with_same_color_nowrap_2(self):
        image = np.array([[6, 8, 8], [8, 6, 8], [8, 8, 6]], dtype=np.uint8)
        actual = count_neighbors_with_same_color_nowrap(image)
        expected = np.array([[1, 3, 2], [3, 2, 3], [2, 3, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_all_neighbors_matching_center_nowrap_1(self):
        image = np.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]], dtype=np.uint8)
        actual = all_neighbors_matching_center_nowrap(image)
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_pixels_with_k_matching_neighbors_nowrap_1(self):
        image = np.array([[9, 7, 9], [7, 7, 7], [9, 7, 9]], dtype=np.uint8)
        actual = pixels_with_k_matching_neighbors_nowrap(image, 3)
        expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_pixels_with_k_matching_neighbors_nowrap_2(self):
        image = np.array([[3, 7, 3, 3], [3, 3, 7, 3], [3, 3, 3, 7]], dtype=np.uint8)
        actual = pixels_with_k_matching_neighbors_nowrap(image, 1)
        expected = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_x_1(self):
        image = np.array([
            [3, 7, 3], 
            [3, 7, 3], 
            [3, 7, 3]], dtype=np.uint8)
        actual = compress_x(image)
        expected = np.array([
            [3, 7, 3], 
            [3, 7, 3], 
            [3, 7, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_x_2(self):
        image = np.array([
            [5, 3, 3, 2, 2, 3], 
            [3, 3, 3, 7, 7, 3], 
            [2, 2, 2, 3, 3, 3]], dtype=np.uint8)
        actual = compress_x(image)
        expected = np.array([
            [5, 3, 2, 3], 
            [3, 3, 7, 3], 
            [2, 2, 3, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_x_3(self):
        image = np.array([
            [1, 2, 2, 2, 3, 4, 4], 
            [7, 7, 7, 7, 7, 7, 7], 
            [1, 2, 2, 2, 3, 4, 4]], dtype=np.uint8)
        actual = compress_x(image)
        expected = np.array([
            [1, 2, 3, 4], 
            [7, 7, 7, 7], 
            [1, 2, 3, 4]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_y_1(self):
        image = np.array([
            [1, 2], 
            [2, 2], 
            [3, 2]], dtype=np.uint8)
        actual = compress_y(image)
        expected = np.array([
            [1, 2], 
            [2, 2], 
            [3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_y_2(self):
        image = np.array([
            [1, 2], 
            [1, 2], 
            [2, 2], 
            [2, 2], 
            [3, 2], 
            [3, 2]], dtype=np.uint8)
        actual = compress_y(image)
        expected = np.array([
            [1, 2], 
            [2, 2], 
            [3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_xy_1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = compress_xy(image)
        expected = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_xy_1(self):
        image = np.array([
            [1, 1, 1, 2, 3, 3], 
            [1, 1, 1, 2, 3, 3], 
            [4, 4, 4, 5, 6, 6], 
            [4, 4, 4, 5, 6, 6], 
            [4, 4, 4, 5, 6, 6], 
            [7, 7, 7, 8, 9, 9]], dtype=np.uint8)
        actual = compress_xy(image)
        expected = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dxplus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, 1, 0)
        expected = np.array([
            [3, 1, 2], 
            [6, 4, 5], 
            [9, 7, 8]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dxminus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, -1, 0)
        expected = np.array([
            [2, 3, 1], 
            [5, 6, 4], 
            [8, 9, 7]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dyplus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, 0, 1)
        expected = np.array([
            [7, 8, 9],
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dyminus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, 0, -1)
        expected = np.array([
            [4, 5, 6], 
            [7, 8, 9],
            [1, 2, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_replace_colors1(self):
        image = np.array([
            [4, 4, 5, 5, 6, 6], 
            [7, 7, 9, 9, 8, 8]], dtype=np.uint8)
        dict = {
            8: 9,
            9: 8,
        }
        actual = image_replace_colors(image, dict)
        expected = np.array([
            [4, 4, 5, 5, 6, 6], 
            [7, 7, 8, 8, 9, 9]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_replace_colors2(self):
        image = np.array([
            [1, 2, 3], 
            [3, 2, 1]], dtype=np.uint8)
        dict = {
            1: 2,
            2: 3,
            3: 1
        }
        actual = image_replace_colors(image, dict)
        expected = np.array([
            [2, 3, 1], 
            [1, 3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_get_row_as_list(self):
        image = np.array([
            [1, 2], 
            [3, 4], 
            [5, 6], 
            [7, 8]], dtype=np.uint8)

        with self.assertRaises(ValueError):
            image_get_row_as_list(image, -1)
        with self.assertRaises(ValueError):
            image_get_row_as_list(image, 4)

        self.assertEqual(image_get_row_as_list(image, 0), [1, 2])
        self.assertEqual(image_get_row_as_list(image, 1), [3, 4])
        self.assertEqual(image_get_row_as_list(image, 2), [5, 6])
        self.assertEqual(image_get_row_as_list(image, 3), [7, 8])

    def test_image_get_column_as_list(self):
        image = np.array([
            [1, 2, 3, 4], 
            [5, 6, 7, 8]], dtype=np.uint8)

        with self.assertRaises(ValueError):
            image_get_column_as_list(image, -1)
        with self.assertRaises(ValueError):
            image_get_column_as_list(image, 4)

        self.assertEqual(image_get_column_as_list(image, 0), [1, 5])
        self.assertEqual(image_get_column_as_list(image, 1), [2, 6])
        self.assertEqual(image_get_column_as_list(image, 2), [3, 7])
        self.assertEqual(image_get_column_as_list(image, 3), [4, 8])

if __name__ == '__main__':
    unittest.main()
