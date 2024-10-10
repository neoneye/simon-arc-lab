import unittest
from .generate_random_values import *

class TestGenerateRandomValues(unittest.TestCase):
    def test_10000_same_min_max(self):
        # Arrange
        g = GenerateRandomValues()
        g.append_value(5, 5)
        g.append_value(2, 2)
        g.append_value(1, 1)
        # Act
        actual = g.find_random_values(0, 100)
        # Assert
        expected = [5, 2, 1]
        self.assertEqual(actual, expected)

    def test_10001_different_min_max(self):
        # Arrange
        g = GenerateRandomValues()
        g.append_value(20, 29)
        g.append_value(10, 19)
        g.append_value(1, 9)
        # Act
        actual = g.find_random_values(0, 100)
        # Assert
        expected = [26, 16, 8]
        self.assertEqual(actual, expected)

    def test_10003_dont_exceed_max_limit(self):
        # Arrange
        g = GenerateRandomValues()
        g.append_value(7, 9)
        g.append_value(4, 6)
        g.append_value(1, 3)
        # Act + Assert
        self.assertEqual(g.find_random_values(0, 16), [8, 5, 2])
        self.assertEqual(g.find_random_values(1, 16), [7, 4, 3])
        self.assertEqual(g.find_random_values(2, 16), [7, 6, 3])
        self.assertEqual(g.find_random_values(3, 16), [7, 5, 1])
        self.assertEqual(g.find_random_values(4, 16), [7, 5, 3])
        self.assertEqual(g.find_random_values(5, 16), [9, 5, 2])
