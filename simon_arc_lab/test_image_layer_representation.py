import unittest
import textwrap
from typing import Dict, Tuple, Optional
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from .histogram import Histogram

class SplitDirection(Enum):
    LR = 'left_right'
    TB = 'top_bottom'

def image_split2_left_right(image: np.array, split: int) -> Tuple[np.array, np.array]:
    height, width = image.shape
    if width < 2 or height < 1:
        raise ValueError("image is too small to split")
    if split < 1 or split >= width:
        raise ValueError("split parameter is outside bounds. The width of the splitted images, must be 1 or more")
    left = image[:, :split]
    right = image[:, split:]
    return left, right

class Node(ABC):
    @abstractmethod
    def print_tree(self, indent: str):
        pass

    @abstractmethod
    def rectangle(self) -> Tuple[int, int, int, int]:
        pass

    @abstractmethod
    def tree_to_string(self, indent: str) -> str:
        pass

    def rectangle_to_compact_string(self) -> str:
        x, y, width, height = self.rectangle()
        return f"{x}_{y}_{width}_{height}"

class SolidNode(Node):
    def __init__(self, x: int, y: int, width: int, height: int, color: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def __str__(self):
        return f"SolidNode x:{self.x} y:{self.y} width:{self.width} height:{self.height} color:{self.color}"

    def __repr__(self):
        return self.__str__()
    
    def print_tree(self, indent: str):
        print(f"{indent}{self}")

    def rectangle(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def tree_to_string(self, indent: str) -> str:
        return f"{indent}{self.rectangle_to_compact_string()} color:{self.color}"

class ImageNode(Node):
    def __init__(self, x: int, y: int, image: np.array, histogram: Histogram):
        self.x = x
        self.y = y
        self.image = image
        self.histogram = histogram
        self.number_of_unique_colors = self.histogram.number_of_unique_colors()

    def __str__(self):
        height, width = self.image.shape
        return f"MultiColorImage x:{self.x} y:{self.y} width:{width} height:{height} number_of_unique_colors:{self.number_of_unique_colors}"

    def __repr__(self):
        return self.__str__()
    
    def print_tree(self, indent: str):
        print(f"{indent}{self}")

    def rectangle(self) -> Tuple[int, int, int, int]:
        height, width = self.image.shape
        return self.x, self.y, width, height

    def tree_to_string(self, indent: str) -> str:
        colors_str = self.histogram.unique_colors_pretty()
        return f"{indent}{self.rectangle_to_compact_string()} image-with-colors:{colors_str}"

class SplitNode(Node):
    def __init__(self, direction: SplitDirection, x: int, y: int, width: int, height: int, size_a: int, size_b: int, score: int, image_a: np.array, image_b: np.array, histogram_a: Histogram, histogram_b: Histogram):
        self.direction = direction
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size_a = size_a
        self.size_b = size_b
        self.score = score
        self.image_a = image_a
        self.image_b = image_b
        self.histogram_a = histogram_a
        self.histogram_b = histogram_b
        self.child_split_a = None
        self.child_split_b = None

    def __str__(self):
        return f"SplitNode direction:{self.direction.name} x:{self.x} y:{self.y} width:{self.width} height:{self.height} size_a:{self.size_a} size_b:{self.size_b} score:{self.score}"

    def __repr__(self):
        return self.__str__()
    
    def print_tree(self, indent: str):
        print(f"{indent}{self}")
        if self.child_split_a is not None:
            self.child_split_a.print_tree(indent + "  ")
        if self.child_split_b is not None:
            self.child_split_b.print_tree(indent + "  ")

    def rectangle(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def tree_to_string(self, indent: str) -> str:
        s = f"{indent}{self.rectangle_to_compact_string()} split:{self.direction.name}"
        if self.child_split_a is not None:
            s += "\n" + self.child_split_a.tree_to_string(indent + ".")
        if self.child_split_b is not None:
            s += "\n" + self.child_split_b.tree_to_string(indent + ".")
        return s

    @staticmethod
    def find_split(direction: SplitDirection, x: int, y: int, image: np.array, verbose: bool) -> Optional['SplitNode']:
        original_height, original_width = image.shape
        if direction == SplitDirection.TB:
            image = image.transpose()

        height, width = image.shape

        found_score = 0
        found_image_a = None
        found_image_b = None
        found_histogram_a = None
        found_histogram_b = None
        found_size_a = None
        found_size_b = None
        # find optimal left-right split determined by score
        for i in range(width-1):
            size_a = i + 1
            image_left, image_right = image_split2_left_right(image, size_a)
            size_b = width - size_a
            assert size_a == image_left.shape[1]
            assert size_b == image_right.shape[1]
            assert size_a + size_b == width
            histogram_a = Histogram.create_with_image(image_left)
            histogram_b = Histogram.create_with_image(image_right)
            color_diff = histogram_a.unique_colors_set() - histogram_b.unique_colors_set()
            if verbose:
                print(f"size_a:{size_a}, size_b:{size_b} a:{histogram_a.number_of_unique_colors()}, b:{histogram_b.number_of_unique_colors()} diff:{color_diff}")
            if len(color_diff) == 0:
                continue
            width_diff = abs(size_a - size_b)
            score = len(color_diff) * (width - width_diff)
            if score <= found_score:
                continue
            if verbose:
                print(f"found better size_a:{size_a}, size_b:{size_b} score:{score}")
            found_score = score
            found_size_a = size_a
            found_size_b = size_b
            found_image_a = image_left
            found_image_b = image_right
            found_histogram_a = histogram_a
            found_histogram_b = histogram_b
        if found_score == 0:
            if verbose:
                print("no candidates found")
            return None
        if verbose:
            print(f"found_score:{found_score}")

        if direction == SplitDirection.TB:
            found_image_a = found_image_a.transpose()
            found_image_b = found_image_b.transpose()

        return SplitNode(
            direction, 
            x, 
            y, 
            original_width, 
            original_height, 
            found_size_a, 
            found_size_b, 
            found_score, 
            found_image_a, 
            found_image_b, 
            found_histogram_a, 
            found_histogram_b
        )

def process_inner(input_x: int, input_y: int, input_image: np.array, input_histogram: Histogram, current_depth: int, max_depth: int, verbose: bool) -> Optional[Node]:
    height, width = input_image.shape
    assert width > 0 and height > 0

    number_of_unique_colors = input_histogram.number_of_unique_colors()
    assert number_of_unique_colors > 0
    if number_of_unique_colors == 1:
        if verbose:
            print("single color. Cannot be split further")
        colors = input_histogram.unique_colors_set()
        assert len(colors) == 1
        solid_color = colors.pop()
        return SolidNode(input_x, input_y, width, height, solid_color)

    if current_depth >= max_depth:
        if verbose:
            print("max depth reached")
        return ImageNode(input_x, input_y, input_image, input_histogram)

    if verbose:
        print("can be split further")
    split_node_lr = SplitNode.find_split(SplitDirection.LR, input_x, input_y, input_image, verbose)
    split_node_tb = SplitNode.find_split(SplitDirection.TB, input_x, input_y, input_image, verbose)

    if verbose:
        print(f"split_node_lr: {split_node_lr}")
        print(f"split_node_tb: {split_node_tb}")
    score_a = 0 if split_node_lr is None else split_node_lr.score
    score_b = 0 if split_node_tb is None else split_node_tb.score
    if score_a == 0 and score_b == 0:
        if verbose:
            print("no candidates found. Not sure how to proceed")
        return ImageNode(input_x, input_y, input_image, input_histogram)
    
    node = None
    split_a_x = input_x
    split_a_y = input_y
    if score_a >= score_b:
        node = split_node_lr
        split_b_x = split_a_x + node.size_a
        split_b_y = split_a_y
    else:
        node = split_node_tb
        split_b_x = split_a_x
        split_b_y = split_a_y + node.size_a
    
    node.child_split_a = process_inner(split_a_x, split_a_y, node.image_a, node.histogram_a, current_depth + 1, max_depth, verbose)
    node.child_split_b = process_inner(split_b_x, split_b_y, node.image_b, node.histogram_b, current_depth + 1, max_depth, verbose)

    return node

def create_nodes_with_image(image: np.array, max_depth: int, verbose: bool) -> Node:
    histogram = Histogram.create_with_image(image)
    x = 0
    y = 0
    node = process_inner(x, y, image, histogram, 0, max_depth, verbose)
    if verbose:
        node.print_tree("")
    return node

def process(image: np.array, max_depth: int, verbose: bool=False) -> str:
    node = create_nodes_with_image(image, max_depth, verbose)
    return node.tree_to_string("|")

class TestImageLayerRepresentation(unittest.TestCase):
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
