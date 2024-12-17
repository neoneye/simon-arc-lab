import unittest
import numpy as np
from .image_sparse_representation import *

class TestImageSparseRepresentation(unittest.TestCase):
    def test_10000_image_to_dictionary_without_size(self):
        image = np.array([
            [7, 7, 9], 
            [8, 7, 9]], dtype=np.uint8)
        actual = image_to_dictionary(image, include_size=False, background_color=None)
        expected = "{(0,0):7,(1,0):7,(2,0):9,(0,1):8,(1,1):7,(2,1):9}"
        self.assertEqual(actual, expected)

    def test_10001_image_to_dictionary_with_size(self):
        image = np.array([
            [0, 1, 2],
            [0, 1, 2]], dtype=np.uint8)
        actual = image_to_dictionary(image, include_size=True, background_color=None)
        expected = "{'width':3,'height':2,(0,0):0,(1,0):1,(2,0):2,(0,1):0,(1,1):1,(2,1):2}"
        self.assertEqual(actual, expected)

    def test_10002_image_to_dictionary_with_background_color(self):
        image = np.array([
            [7, 7, 9],
            [8, 7, 9]], dtype=np.uint8)
        actual = image_to_dictionary(image, include_size=False, background_color=7)
        expected = "{'background':7,(2,0):9,(0,1):8,(2,1):9}"
        self.assertEqual(actual, expected)

    def test_20000_dictionary_to_image_in_order(self):
        input_str = "{'width':3,'height':2,(0,0):7,(1,0):7,(2,0):9,(0,1):8,(1,1):7,(2,1):9}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual_image, expected))
        self.assertIsNone(actual_status)

    def test_20001_dictionary_to_image_out_of_order(self):
        input_str = "{(0,1):8,(1,1):7,(2,1):9,(0,0):7,(1,0):7,(2,0):9,'height':2,'width':3}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual_image, expected))
        self.assertIsNone(actual_status)

    def test_20002_dictionary_to_image_with_newlines(self):
        input_str = "{'width':3,'height':2,\n(0,0):7,(1,0):7,(2,0):9,(0,1):8,(1,1):7,\n(2,1):9}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual_image, expected))
        self.assertIsNone(actual_status)

    def test_20003_dictionary_to_image_with_spaces(self):
        input_str = "{'width':3,'height': 2,\n(0, 0): 7, (1 ,0):7,(2,0):9,(0,1):8,(1,1):7,\n(2,1):9}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual_image, expected))
        self.assertIsNone(actual_status)
