import unittest
import numpy as np
from .image_pattern import *

class TestImagePattern(unittest.TestCase):
    def test_10000_checkerboard_colors2_size1_offset00(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 1, 0, 0, [5, 6])
        # Assert
        expected = np.array([
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6],
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_checkerboard_colors2_size2_offset00(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 2, 0, 0, [0, 1])
        # Assert
        expected = np.array([
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_checkerboard_colors2_size2_offset10(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 2, 1, 0, [0, 1])
        # Assert
        expected = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_checkerboard_colors2_size2_offset01(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 2, 0, 1, [0, 1])
        # Assert
        expected = np.array([
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10004_checkerboard_colors2_size3_offset00(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 3, 0, 0, [0, 1])
        # Assert
        expected = np.array([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10005_checkerboard_colors3_size2_offset00(self):
        # Act
        actual = image_pattern_checkerboard(7, 5, 2, 0, 0, [0, 1, 2])
        # Assert
        expected = np.array([
            [0, 0, 1, 1, 2, 2, 0],
            [0, 0, 1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0, 0, 1],
            [1, 1, 2, 2, 0, 0, 1],
            [2, 2, 0, 0, 1, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_lines_horizontal_colors2_size1_offset0(self):
        # Act
        actual = image_pattern_lines_horizontal(3, 5, 1, 0, [5, 6])
        # Assert
        expected = np.array([
            [5, 5, 5],
            [6, 6, 6],
            [5, 5, 5],
            [6, 6, 6],
            [5, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20001_lines_horizontal_colors2_size2_offset0(self):
        # Act
        actual = image_pattern_lines_horizontal(3, 5, 2, 0, [5, 6])
        # Assert
        expected = np.array([
            [5, 5, 5],
            [5, 5, 5],
            [6, 6, 6],
            [6, 6, 6],
            [5, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20002_lines_horizontal_colors3_size2_offset0(self):
        # Act
        actual = image_pattern_lines_horizontal(3, 5, 2, 0, [5, 6, 7])
        # Assert
        expected = np.array([
            [5, 5, 5],
            [5, 5, 5],
            [6, 6, 6],
            [6, 6, 6],
            [7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20003_lines_horizontal_colors3_size2_offset1(self):
        # Act
        actual = image_pattern_lines_horizontal(3, 5, 2, 1, [5, 6, 7])
        # Assert
        expected = np.array([
            [5, 5, 5],
            [6, 6, 6],
            [6, 6, 6],
            [7, 7, 7],
            [7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30000_lines_vertical_colors2_size1_offset0(self):
        # Act
        actual = image_pattern_lines_vertical(3, 5, 1, 0, [5, 6])
        # Assert
        expected = np.array([
            [5, 6, 5],
            [5, 6, 5],
            [5, 6, 5],
            [5, 6, 5],
            [5, 6, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30001_lines_vertical_colors2_size2_offset0(self):
        # Act
        actual = image_pattern_lines_vertical(5, 4, 2, 0, [5, 6])
        # Assert
        expected = np.array([
            [5, 5, 6, 6, 5],
            [5, 5, 6, 6, 5],
            [5, 5, 6, 6, 5],
            [5, 5, 6, 6, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30002_lines_vertical_colors2_size2_offset1(self):
        # Act
        actual = image_pattern_lines_vertical(5, 4, 2, 1, [5, 6])
        # Assert
        expected = np.array([
            [5, 6, 6, 5, 5],
            [5, 6, 6, 5, 5],
            [5, 6, 6, 5, 5],
            [5, 6, 6, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40000_image_pattern_lines_slope45degree_colors2(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, 1, 1, [5, 6])
        # Assert
        expected = np.array([
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6],
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40001_image_pattern_lines_plus1plus2_colors2(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, 2, 1, [5, 6])
        # Assert
        expected = np.array([
            [5, 5, 6, 6, 5],
            [6, 6, 5, 5, 6],
            [5, 5, 6, 6, 5],
            [6, 6, 5, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40002_image_pattern_lines_plus3plus1_colors2(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, 3, 1, [5, 6])
        # Assert
        expected = np.array([
            [5, 5, 5, 6, 6],
            [6, 6, 6, 5, 5],
            [5, 5, 5, 6, 6],
            [6, 6, 6, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40003_image_pattern_lines_plus1plus3_colors2(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, 1, 3, [5, 6])
        # Assert
        expected = np.array([
            [5, 6, 5, 6, 5],
            [5, 6, 5, 6, 5],
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40004_image_pattern_lines_plus2minus1_colors3(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, 2, -1, [5, 6, 7])
        # Assert
        expected = np.array([
            [5, 5, 6, 6, 7],
            [6, 6, 7, 7, 5],
            [7, 7, 5, 5, 6],
            [5, 5, 6, 6, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40005_image_pattern_lines_minus1plus2_colors3(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, -1, 2, [5, 6, 7])
        # Assert
        expected = np.array([
            [5, 6, 7, 5, 6],
            [5, 6, 7, 5, 6],
            [6, 7, 5, 6, 7],
            [6, 7, 5, 6, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40006_image_pattern_lines_minus1minus1_colors3(self):
        # Act
        actual = image_pattern_lines_slope(5, 4, -1, -1, [5, 6, 7])
        # Assert
        expected = np.array([
            [5, 6, 7, 5, 6],
            [6, 7, 5, 6, 7],
            [7, 5, 6, 7, 5],
            [5, 6, 7, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

