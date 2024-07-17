import unittest
import numpy as np
from .game_of_life import *

class TestGameOfLife(unittest.TestCase):
    def test_blinker(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = game_of_life_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 1, 1, 1, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_glider(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0], 
            [0, 1, 0, 1, 0], 
            [0, 0, 1, 1, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = game_of_life_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1], 
            [0, 0, 1, 1, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
