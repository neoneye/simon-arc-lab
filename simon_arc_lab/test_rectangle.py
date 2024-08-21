import unittest
import numpy as np
from .rectangle import *

class TestRectangle(unittest.TestCase):
    def test_10000_intersection_full(self):
        # Arrange
        a = Rectangle(2, 2, 20, 4)
        b = Rectangle(2, 2, 20, 4)
        # Act
        actual = a.intersection(b)
        # Assert
        expected = Rectangle(2, 2, 20, 4)
        self.assertEqual(actual, expected)

    def test_10001_intersection_some(self):
        # Arrange
        a = Rectangle(0, 0, 10, 10)
        b = Rectangle(2, 2, 20, 4)
        # Act
        actual = a.intersection(b)
        # Assert
        expected = Rectangle(2, 2, 8, 4)
        self.assertEqual(actual, expected)

    def test_10002_intersection_none(self):
        a = Rectangle(0, 0, 10, 10)
        b = Rectangle(10, 0, 10, 10)
        # Act
        actual = a.intersection(b)
        # Assert
        expected = Rectangle.empty()
        self.assertEqual(actual, expected)
