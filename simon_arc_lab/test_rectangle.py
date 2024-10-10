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

    def test_10003_has_overlap_true(self):
        a = Rectangle(5, 5, 10, 10)
        # Act
        actual = a.has_overlap(a)
        # Assert
        self.assertEqual(actual, True)

    def test_10004_has_overlap_true(self):
        a = Rectangle(0, 0, 10, 10)
        b = Rectangle(9, 9, 10, 10)
        # Act
        actual = a.has_overlap(b)
        # Assert
        self.assertEqual(actual, True)

    def test_10005_has_overlap_false(self):
        a = Rectangle(10, 9, 10, 10)
        b = Rectangle(0, 0, 10, 10)
        # Act
        actual = a.has_overlap(b)
        # Assert
        self.assertEqual(actual, False)

    def test_10006_has_overlap_false(self):
        a = Rectangle(11, 0, 10, 10)
        b = Rectangle(0, 0, 10, 10)
        # Act
        actual = a.has_overlap(b)
        # Assert
        self.assertEqual(actual, False)

    def test_20000_random_child_rectangle(self):
        # Arrange
        a = Rectangle(0, 100, 10, 10)
        # Act
        actual = a.random_child_rectangle(0)
        # Assert
        expected = Rectangle(3, 103, 3, 1)
        self.assertEqual(actual, expected)

    def test_20001_random_child_rectangle(self):
        # Arrange
        a = Rectangle(0, 100, 10, 10)
        # Act
        actual = a.random_child_rectangle(1)
        # Assert
        expected = Rectangle(3, 104, 1, 4)
        self.assertEqual(actual, expected)

    def test_30000_mass(self):
        self.assertEqual(Rectangle(1, 2, 3, 4).mass(), 12)
        self.assertEqual(Rectangle(100, 200, 5, 5).mass(), 25)
        self.assertEqual(Rectangle(0, 0, 0, 10).mass(), 0)
        self.assertEqual(Rectangle(0, 0, 10, 0).mass(), 0)
        self.assertEqual(Rectangle(0, 0, -10, -10).mass(), 0)
