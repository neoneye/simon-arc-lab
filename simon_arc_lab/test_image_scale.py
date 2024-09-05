import unittest
import numpy as np
from .image_scale import image_scale, image_scale_up_variable

class TestImageScale(unittest.TestCase):
    def test_scale_do_nothing(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        input_image, output_image = image_scale(image, 'up', 1, 'up', 1)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(input_image, expected)
        np.testing.assert_array_equal(output_image, expected)

    def test_scale_x_up(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        input_image, output_image = image_scale(image, 'up', 2, 'up', 1)
        # Assert
        expected_input = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        expected_output = np.array([
            [1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(input_image, expected_input)
        np.testing.assert_array_equal(output_image, expected_output)

    def test_scale_x_down(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        input_image, output_image = image_scale(image, 'down', 2, 'up', 1)
        # Assert
        expected_input = np.array([
            [1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6]], dtype=np.uint8)
        expected_output = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(input_image, expected_input)
        np.testing.assert_array_equal(output_image, expected_output)

    def test_scale_y_up(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        input_image, output_image = image_scale(image, 'up', 1, 'up', 2)
        # Assert
        expected_input = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        expected_output = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(input_image, expected_input)
        np.testing.assert_array_equal(output_image, expected_output)

    def test_scale_y_down(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        input_image, output_image = image_scale(image, 'up', 1, 'down', 2)
        # Assert
        expected_input = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 6]], dtype=np.uint8)
        expected_output = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(input_image, expected_input)
        np.testing.assert_array_equal(output_image, expected_output)

    def test_image_scale_up_variable_one_or_more(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        xs = [2, 1, 2]
        ys = [3, 3]
        # Act
        actual = image_scale_up_variable(image, xs, ys)
        # Assert
        expected = np.array([
            [1, 1, 2, 3, 3],
            [1, 1, 2, 3, 3],
            [1, 1, 2, 3, 3],
            [4, 4, 5, 6, 6],
            [4, 4, 5, 6, 6],
            [4, 4, 5, 6, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_image_scale_up_variable_zeros(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        xs = [1, 0, 1]
        ys = [1, 1]
        # Act
        actual = image_scale_up_variable(image, xs, ys)
        # Assert
        expected = np.array([
            [1, 3],
            [4, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

