import unittest
import numpy as np
from .image_string_representation import *

class TestImageStringRepresentation(unittest.TestCase):
    def test_10000_image_to_string(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        actual = image_to_string(image)
        expected = "123\n456"
        self.assertEqual(actual, expected)

    def test_10001_image_from_string(self):
        image = "123\n456"
        actual = image_from_string(image)
        expected = np.array([
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_to_string_long_lowercase_colornames(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_long_lowercase_colornames(image)
        expected = "black blue red green\nyellow grey purple orange\ncyan brown white white"
        self.assertEqual(actual, expected)

    def test_30000_image_to_string_emoji_circles_v1(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_emoji_circles_v1(image)
        expected = "⚫🔵🔴🟢\n🟡⚪🟣🟠\n🟦🟤❌❌"
        self.assertEqual(actual, expected)

    def test_40000_image_to_string_emoji_chess_v1(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_emoji_chess_v1(image)
        expected = "♔♕♖♗\n♘♙♚♛\n♜♝♞♞"
        self.assertEqual(actual, expected)
