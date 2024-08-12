import unittest
import numpy as np
from .image_scale import image_scale

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

