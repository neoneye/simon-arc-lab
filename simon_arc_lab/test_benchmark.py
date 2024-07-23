import unittest
import numpy as np
from .benchmark import *

class TestBenchmark(unittest.TestCase):
    def test_image_size1d_to_string(self):
        self.assertEqual(image_size1d_to_string(0), 'small')
        self.assertEqual(image_size1d_to_string(10), 'small')
        self.assertEqual(image_size1d_to_string(11), 'medium')
        self.assertEqual(image_size1d_to_string(20), 'medium')
        self.assertEqual(image_size1d_to_string(21), 'large')
        self.assertEqual(image_size1d_to_string(30), 'large')
        self.assertEqual(image_size1d_to_string(31), 'other')
