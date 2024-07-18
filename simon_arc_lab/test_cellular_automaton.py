import unittest
import numpy as np
from .cellular_automaton import *

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

    def test_wireworld_iteration1(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 3, 3, 0, 0, 0], 
            [3, 2, 1, 3, 0, 3, 3, 0], 
            [0, 0, 0, 3, 3, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_wireworld_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 3, 0, 0, 0], 
            [3, 3, 2, 1, 0, 3, 3, 0], 
            [0, 0, 0, 1, 3, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_wireworld_iteration2(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 3, 0, 0, 0], 
            [3, 3, 2, 1, 0, 3, 3, 0], 
            [0, 0, 0, 1, 3, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_wireworld_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 2, 1, 0, 0, 0], 
            [3, 3, 3, 2, 0, 3, 3, 0], 
            [0, 0, 0, 2, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_wireworld_iteration3(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 2, 1, 0, 0, 0], 
            [3, 3, 3, 2, 0, 3, 3, 0], 
            [0, 0, 0, 2, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_wireworld_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 3, 2, 0, 0, 0], 
            [3, 3, 3, 3, 0, 1, 3, 0], 
            [0, 0, 0, 3, 2, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_wireworld_iteration4(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 3, 2, 0, 0, 0], 
            [3, 3, 3, 3, 0, 1, 3, 0], 
            [0, 0, 0, 3, 2, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_wireworld_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 3, 3, 0, 0, 0], 
            [3, 3, 3, 3, 0, 2, 1, 0], 
            [0, 0, 0, 3, 3, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_cave_iteration0(self):
        input = np.array([
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_cave_wrap(input)
        expected = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_maze_iteration0(self):
        input = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual = cellular_automata_maze_wrap(input)
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
