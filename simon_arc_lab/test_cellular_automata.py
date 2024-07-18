import unittest
import numpy as np
from .cellular_automata import *

class TestCellularAutomata(unittest.TestCase):
    def test_gameoflife_blinker(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_gameoflife_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 1, 1, 1, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_gameoflife_glider(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0], 
            [0, 1, 0, 1, 0], 
            [0, 0, 1, 1, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_gameoflife_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1], 
            [0, 0, 1, 1, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_gameoflife_alive6(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 0], 
            [0, 1, 0, 1, 0], 
            [0, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_gameoflife_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 0], 
            [0, 1, 0, 1, 0], 
            [0, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_highlife_blinker(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_highlife_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 1, 1, 1, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_highlife_alive6(self):
        input = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 0], 
            [0, 1, 0, 1, 0], 
            [0, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_highlife_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 0], 
            [0, 1, 1, 1, 0], 
            [0, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_serviettes_iteration1(self):
        input = np.array([
            [0, 0, 0, 0], 
            [0, 1, 1, 0], 
            [0, 1, 1, 0], 
            [0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_serviettes_wrap(input)
        expected = np.array([
            [0, 1, 1, 0], 
            [1, 0, 0, 1], 
            [1, 0, 0, 1],
            [0, 1, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_serviettes_iteration2(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 0, 0], 
            [0, 1, 0, 0, 1, 0], 
            [0, 1, 0, 0, 1, 0], 
            [0, 0, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_serviettes_wrap(input)
        expected = np.array([
            [0, 0, 1, 1, 0, 0], 
            [0, 1, 0, 0, 1, 0], 
            [1, 0, 1, 1, 0, 1], 
            [1, 0, 1, 1, 0, 1], 
            [0, 1, 0, 0, 1, 0], 
            [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_serviettes_iteration3(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 1, 0, 1, 1, 0, 1, 0], 
            [0, 1, 0, 1, 1, 0, 1, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_serviettes_wrap(input)
        expected = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 1, 0, 0, 0, 0, 1, 0], 
            [1, 0, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 0, 0, 1], 
            [0, 1, 0, 0, 0, 0, 1, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
