import unittest
import numpy as np
from .image_object_mass import *

class TestImageObjectMass(unittest.TestCase):
    def test_10000_object_mass_ok(self):
        # Arrange
        mask_a = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        mask_b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        mask_c = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.uint8)

        input_objects = [mask_a, mask_b, mask_c]

        # Act
        output = object_mass(input_objects)

        # Assert
        a = 6
        b = 4
        c = 1
        expected_pixels = np.array([
            [a, a, 0, 0, 0],
            [a, a, b, b, 0],
            [a, a, b, b, 0],
            [0, 0, 0, 0, c],
        ], dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_pixels)

    def test_10001_object_enumerate_exceed_maximum(self):
        # Arrange
        input_objects = [np.zeros((1, 1), dtype=np.uint8) for _ in range(256)]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_mass(input_objects)
        self.assertTrue("maximum" in str(context.exception))

    def test_10002_object_enumerate_different_sizes(self):
        # Arrange
        input_objects = [
            np.zeros((1, 2), dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8)
        ]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_mass(input_objects)
        self.assertTrue("same size" in str(context.exception))

    def test_10003_object_enumerate_too_small(self):
        # Arrange
        input_objects = [np.zeros((0, 0), dtype=np.uint8)]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_mass(input_objects)
        self.assertTrue("1x1 or bigger" in str(context.exception))

    def test_10004_object_enumerate_invalid_mask_color(self):
        # Arrange
        input_objects = [np.array([[5]], dtype=np.uint8)]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_mass(input_objects)
        self.assertTrue("Invalid mask" in str(context.exception))

    def test_10005_object_enumerate_overlap_detected(self):
        # Arrange
        pixels0 = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        pixels1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ], dtype=np.uint8)

        input_objects = [pixels0, pixels1]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_mass(input_objects)
        self.assertTrue("Detected overlap" in str(context.exception))

    def test_10006_object_mass_clamp(self):
        # Arrange
        # create image with size 16x16 and fill it with 1, so the mass is 256
        mask = np.ones((16, 16), dtype=np.uint8)

        input_objects = [mask]

        # Act
        output = object_mass(input_objects)

        # Assert
        a = 255 # however since the mass is greater than what can be stored in a uint8, it gets clamped to 255
        expected_pixels = np.full((16, 16), a, dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_pixels)

if __name__ == "__main__":
    unittest.main()
