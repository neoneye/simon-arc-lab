import unittest
import numpy as np
from .image_object_enumerate import *

class TestImageObjectEnumerate(unittest.TestCase):
    def test_10000_object_enumerate_ok(self):
        # Arrange
        pixels0 = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        pixels1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        pixels2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.uint8)

        input_objects = [pixels0, pixels1, pixels2]

        # Act
        output = object_enumerate(input_objects)

        # Assert
        expected_pixels = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [0, 0, 0, 0, 3],
        ], dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_pixels)

    def test_10001_object_enumerate_exceed_maximum(self):
        # Arrange
        input_objects = [np.zeros((1, 1), dtype=np.uint8) for _ in range(256)]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_enumerate(input_objects)
        self.assertTrue("maximum" in str(context.exception))

    def test_10002_object_enumerate_different_sizes(self):
        # Arrange
        input_objects = [
            np.zeros((1, 2), dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8)
        ]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_enumerate(input_objects)
        self.assertTrue("same size" in str(context.exception))

    def test_10003_object_enumerate_too_small(self):
        # Arrange
        input_objects = [np.zeros((0, 0), dtype=np.uint8)]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_enumerate(input_objects)
        self.assertTrue("1x1 or bigger" in str(context.exception))

    def test_10004_object_enumerate_invalid_mask_color(self):
        # Arrange
        input_objects = [np.array([[5]], dtype=np.uint8)]

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            object_enumerate(input_objects)
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
            object_enumerate(input_objects)
        self.assertTrue("Detected overlap" in str(context.exception))

if __name__ == "__main__":
    unittest.main()
