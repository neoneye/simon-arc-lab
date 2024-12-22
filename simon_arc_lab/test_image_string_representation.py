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

    def test_11000_image_to_string_spaces(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        actual = image_to_string_spaces(image)
        expected = "1 2 3\n4 5 6"
        self.assertEqual(actual, expected)

    def test_20000_image_to_string_colorname(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_colorname(image)
        expected = "black blue red green\nyellow grey purple orange\ncyan brown white white"
        self.assertEqual(actual, expected)

    def test_30000_spreadsheet_column_name(self):
        self.assertEqual(ImageToString.spreadsheet_column_name(0), 'A')
        self.assertEqual(ImageToString.spreadsheet_column_name(25), 'Z')
        self.assertEqual(ImageToString.spreadsheet_column_name(26), 'AA')
        self.assertEqual(ImageToString.spreadsheet_column_name(51), 'AZ')
        self.assertEqual(ImageToString.spreadsheet_column_name(52), 'BA')

    def test_30001_image_to_string_spreadsheet_v1(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_spreadsheet_v1(image)
        expected = ",A,B,C,D\n1,0,1,2,3\n2,4,5,6,7\n3,8,9,10,11"
        self.assertEqual(actual, expected)

    def test_40000_image_to_string_emoji_circles_v1(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_emoji_circles_v1(image)
        expected = "⚫🔵🔴🟢\n🟡⚪🟣🟠\n🟦🟤❌❌"
        self.assertEqual(actual, expected)

    def test_40001_image_to_string_emoji_chess_without_indices_v1(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_emoji_chess_without_indices_v1(image)
        expected = "♔♕♖♗\n♘♙♚♛\n♜♝♞♞"
        self.assertEqual(actual, expected)

    def test_40002_image_to_string_emoji_chess_with_indices_v1(self):
        image = np.array([
            [0, 1, 2, 3], 
            [4, 5, 6, 7],
            [8, 9, 10, 11]], dtype=np.uint8)
        actual = image_to_string_emoji_chess_with_indices_v1(image)
        expected = "3♔♕♖♗\n2♘♙♚♛\n1♜♝♞♞\n ABCD"
        self.assertEqual(actual, expected)
