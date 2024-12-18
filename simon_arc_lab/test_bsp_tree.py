import unittest
import textwrap
import numpy as np
from .bsp_tree import *

def process(image: np.array, max_depth: int, verbose: bool=False) -> str:
    node = create_bsp_tree(image, max_depth, verbose)
    recreated_image = node.to_image()
    if np.array_equal(image, recreated_image) == False:
        raise Exception("The recreated image is not equal to the original image.")
    return node.tree_to_string("|")

class TestBSPTree(unittest.TestCase):
    def test_10000_solid_color(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        actual = process(image, 3)

        expected = textwrap.dedent("""
        |0_0_5_3 color:5
        """).strip()
        self.assertEqual(actual, expected)

    def test_11000_direction_topbottom(self):
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

    def test_12000_direction_leftright(self):
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

    def test_13000_direction_leftright_and_direction_topbottom(self):
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

    def test_20000_small_image_with_one_pixel_in_the_center(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        image[1, 1] = 42
        actual = process(image, 4)

        expected = textwrap.dedent("""
        |0_0_3_3 split:LR
        |.0_0_2_3 split:TB
        |..0_0_2_2 split:LR
        |...0_0_1_2 color:0
        |...1_0_1_2 split:TB
        |....1_0_1_1 color:0
        |....1_1_1_1 color:42
        |..0_2_2_1 color:0
        |.2_0_1_3 color:0
        """).strip()
        self.assertEqual(actual, expected)

    def test_20001_big_image_with_one_pixel_in_the_center(self):
        image = np.zeros((99, 51), dtype=np.uint8)
        image[50, 25] = 1
        actual = process(image, 8)

        expected = textwrap.dedent("""
        |0_0_51_99 split:TB
        |.0_0_51_51 split:LR
        |..0_0_26_51 split:LR
        |...0_0_13_51 color:0
        |...13_0_13_51 split:LR
        |....13_0_6_51 color:0
        |....19_0_7_51 split:LR
        |.....19_0_3_51 color:0
        |.....22_0_4_51 split:LR
        |......22_0_2_51 color:0
        |......24_0_2_51 split:LR
        |.......24_0_1_51 color:0
        |.......25_0_1_51 split:TB
        |........25_0_1_50 color:0
        |........25_50_1_1 color:1
        |..26_0_25_51 color:0
        |.0_51_51_48 color:0
        """).strip()
        self.assertEqual(actual, expected)
