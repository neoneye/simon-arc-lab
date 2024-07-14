import unittest
import numpy as np
from .game_of_life import *

class TestGameOfLife(unittest.TestCase):
    def test_blinker(self):
        input = np.zeros((5, 5), dtype=np.uint8)
        input[1, 2] = 1
        input[2, 2] = 1
        input[3, 2] = 1
        actual = game_of_life_wrap(input)
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[2, 1] = 1
        expected[2, 2] = 1
        expected[2, 3] = 1
        self.assertTrue(np.array_equal(actual, expected))

    def test_glider(self):
        input = np.zeros((5, 5), dtype=np.uint8)
        input[1, 3] = 1
        input[2, 1] = 1
        input[2, 3] = 1
        input[3, 2] = 1
        input[3, 3] = 1
        actual = game_of_life_wrap(input)
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1, 2] = 1
        expected[2, 3] = 1
        expected[2, 4] = 1
        expected[3, 2] = 1
        expected[3, 3] = 1
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
