import unittest
import numpy as np
from .python_image_builder import PythonImageBuilder

class TestPythonImageBuilder(unittest.TestCase):
    def test_10000_zeros_and_rows(self):
        # Arrange
        original_image = np.array([
            [0, 0, 5, 5, 0, 0],
            [0, 0, 5, 5, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        builder = PythonImageBuilder(original_image, 0, name="image")
        builder.rectangle(2, 0, 2, 2, 5)
        builder.rectangle(0, 2, 6, 1, 1)
        builder.rectangle(0, 4, 6, 2, 3)
        # Act
        actual = builder.lines
        # Assert
        expected = [
            "image=np.zeros((7,6),dtype=np.uint8)",
            "image[0:2,2:4]=5",
            "image[2,:]=1",
            "image[4:6,:]=3",
        ]
        self.assertEqual(actual, expected)

    def test_10001_full_and_columns(self):
        # Arrange
        original_image = np.array([
            [9, 9, 5, 5, 9, 9],
            [9, 9, 5, 5, 9, 9],
            [1, 1, 5, 5, 1, 1],
            [9, 9, 5, 5, 9, 9],
            [3, 3, 5, 5, 3, 3],
            [3, 3, 5, 5, 3, 3],
            [9, 9, 5, 5, 9, 9]], dtype=np.uint8)
        builder = PythonImageBuilder(original_image, 9, name="image")
        builder.rectangle(2, 0, 2, 7, 5)
        builder.rectangle(0, 2, 6, 2, 1)
        builder.rectangle(0, 4, 2, 2, 3)
        builder.rectangle(4, 4, 2, 2, 3)
        # Act
        actual = builder.lines
        # Assert
        expected = [
            "image=np.full((7,6),9,dtype=np.uint8)",
            "image[:,2:4]=5",
            "image[2:4,:]=1",
            "image[4:6,0:2]=3",
            "image[4:6,4:6]=3"
        ]
        self.assertEqual(actual, expected)
