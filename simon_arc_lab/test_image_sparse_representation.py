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
            [8, 7, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertIsNone(actual_status)

    def test_20001_dictionary_to_image_out_of_order(self):
        input_str = "{(0,1):8,(1,1):7,(2,1):9,(0,0):7,(1,0):7,(2,0):9,'height':2,'width':3}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertIsNone(actual_status)

    def test_20002_dictionary_to_image_with_newlines(self):
        input_str = "{'width':3,'height':2,\n(0,0):7,(1,0):7,(2,0):9,(0,1):8,(1,1):7,\n(2,1):9}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertIsNone(actual_status)

    def test_20003_dictionary_to_image_with_spaces(self):
        input_str = "{'width':3,'height': 2,\n(0, 0): 7, (1 ,0):7,(2,0):9,(0,1):8,(1,1):7,\n(2,1):9}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [7, 7, 9],
            [8, 7, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertIsNone(actual_status)

    def test_20004_dictionary_to_image_with_background_a(self):
        input_str = "{'width':2,'height':4,'background':42}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [42, 42],
            [42, 42],
            [42, 42],
            [42, 42]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertIsNone(actual_status)

    def test_20005_dictionary_to_image_with_background_b(self):
        input_str = "{'width':3,'height':3,'background':0, (1,1):  99}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [0, 0, 0],
            [0, 99, 0],
            [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertIsNone(actual_status)

    def test_20100_dictionary_to_image_problem_pixel_outside(self):
        input_str = "{'width':3,'height':3,'background':0, (100,5):  99}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertEqual(actual_status, "1 pixels outside")

    def test_20101_dictionary_to_image_problem_unassigned(self):
        input_str = "{'width':1,'height':1}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [255]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertEqual(actual_status, "1 unassigned pixels")

    def test_20102_dictionary_to_image_problem_unassigned_and_outside(self):
        input_str = "{'width':1,'height':1,(100,5):99}}"
        actual_image, actual_status = dictionary_to_image(input_str)
        expected = np.array([
            [255]], dtype=np.uint8)
        np.testing.assert_array_equal(actual_image, expected)
        self.assertEqual(actual_status, "1 pixels outside, 1 unassigned pixels")

    def test_20200_dictionary_to_image_width_is_missing(self):
        input_str = "{'wi_typo_dth':1,'height':1,(0,0):42}}"
        with self.assertRaises(ValueError) as context:
            dictionary_to_image(input_str)
        self.assertEqual(str(context.exception), "Missing 'width'")

    def test_20201_dictionary_to_image_height_is_missing(self):
        input_str = "{'width':1,'hei_typo_ght':1,(0,0):42}}"
        with self.assertRaises(ValueError) as context:
            dictionary_to_image(input_str)
        self.assertEqual(str(context.exception), "Missing 'height'")
