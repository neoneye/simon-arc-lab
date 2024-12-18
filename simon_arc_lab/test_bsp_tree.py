import unittest
import textwrap
import numpy as np
from .histogram import Histogram
from .bsp_tree import *

def process(image: np.array, max_depth: int, verbose: bool=False) -> str:
    node = create_bsp_tree(image, max_depth, verbose)
    return node.tree_to_string("|")

class TestBSPTree(unittest.TestCase):
    def test_10000_direction_topbottom(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        actual = process(image, 3)

        expected = textwrap.dedent("""
        |0_0_5_4 split:TB
        |.0_0_5_2 color:5
        |.0_2_5_2 split:TB
        |..0_2_5_1 color:3
        |..0_3_5_1 color:1
        """).strip()
        self.assertEqual(actual, expected)

    def test_11000_direction_leftright(self):
        image = np.array([
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1]], dtype=np.uint8)
        actual = process(image, 3)

        expected = textwrap.dedent("""
        |0_0_4_5 split:LR
        |.0_0_2_5 color:5
        |.2_0_2_5 split:LR
        |..2_0_1_5 color:3
        |..3_0_1_5 color:1
        """).strip()
        self.assertEqual(actual, expected)

    def test_11000_direction_leftright_and_direction_topbottom(self):
        image = np.array([
            [5, 5, 3, 1],
            [7, 7, 7, 7],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1]], dtype=np.uint8)
        actual = process(image, 4)

        expected = textwrap.dedent("""
        |0_0_4_5 split:LR
        |.0_0_2_5 split:TB
        |..0_0_2_2 split:TB
        |...0_0_2_1 color:5
        |...0_1_2_1 color:7
        |..0_2_2_3 color:5
        |.2_0_2_5 split:TB
        |..2_0_2_2 split:TB
        |...2_0_2_1 split:LR
        |....2_0_1_1 color:3
        |....3_0_1_1 color:1
        |...2_1_2_1 color:7
        |..2_2_2_3 split:LR
        |...2_2_1_3 color:3
        |...3_2_1_3 color:1
        """).strip()
        self.assertEqual(actual, expected)
