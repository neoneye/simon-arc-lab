import unittest
from typing import Dict, Tuple, Optional
import numpy as np
from enum import Enum
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

class Split2:
    def __init__(self, direction: SplitDirection, size_a: int, size_b: int, score: int, image_a: np.array, image_b: np.array, histogram_a: Histogram, histogram_b: Histogram):
        self.direction = direction
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
        return f"{self.direction.name} size_a:{self.size_a}, size_b:{self.size_b} score:{self.score}"

    def __repr__(self):
        return self.__str__()
    
    def print_tree(self, indent: str):
        print(f"{indent}{self}")
        if self.child_split_a is not None:
            self.child_split_a.print_tree(indent + "  ")
        if self.child_split_b is not None:
            self.child_split_b.print_tree(indent + "  ")

    @staticmethod
    def find_split(direction: SplitDirection, image: np.array) -> Optional['Split2']:
        verbose = True
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
        return Split2(direction, found_size_a, found_size_b, found_score, found_image_a, found_image_b, found_histogram_a, found_histogram_b)
    

def process_inner(image: np.array, current_depth: int, max_depth: int) -> Optional[Split2]:
    if current_depth >= max_depth:
        return None
    height, width = image.shape

    candidate_a = Split2.find_split(SplitDirection.LR, image)
    image_transposed = image.transpose()
    candidate_b = Split2.find_split(SplitDirection.TB, image_transposed)

    print(f"candidate_a:{candidate_a}")
    print(f"candidate_b:{candidate_b}")
    score_a = 0 if candidate_a is None else candidate_a.score
    score_b = 0 if candidate_b is None else candidate_b.score
    if score_a == 0 and score_b == 0:
        return None
    candidate = None
    if score_a >= score_b:
        candidate = candidate_a
    else:
        candidate_b.image_a = candidate_b.image_a.transpose()
        candidate_b.image_b = candidate_b.image_b.transpose()
        candidate = candidate_b
    
    if candidate.histogram_a.number_of_unique_colors() > 1:
        print("A can be split further")
        candidate.child_split_a = process_inner(candidate.image_a, current_depth + 1, max_depth)
    else:
        print("A is a single color")

    if candidate.histogram_b.number_of_unique_colors() > 1:
        print("B can be split further")
        candidate.child_split_b = process_inner(candidate.image_b, current_depth + 1, max_depth)
    else:
        print("B is a single color")

    return candidate

def process(image: np.array) -> str:
    print("----------- simon is testing ---------")
    height, width = image.shape

    candidate = process_inner(image, 0, 3)
    candidate.print_tree("")

    # candidate_a = Split2.find_split(SplitDirection.LR, image)
    # image_transposed = image.transpose()
    # candidate_b = Split2.find_split(SplitDirection.TB, image_transposed)

    # print(f"candidate_a:{candidate_a}")
    # print(f"candidate_b:{candidate_b}")
    # score_a = 0 if candidate_a is None else candidate_a.score
    # score_b = 0 if candidate_b is None else candidate_b.score
    # if score_a == 0 and score_b == 0:
    #     return 'none'
    # if score_a > score_b:
    #     if candidate_a.histogram_left.number_of_unique_colors() > 1:
    #         print("left can be split further")
    #     else:
    #         print("left is a single color")

    #     if candidate_a.histogram_right.number_of_unique_colors() > 1:
    #         print("right can be split further")
    #     else:
    #         print("right is a single color")

    #     return 'left|right'

    # if candidate_b.histogram_left.number_of_unique_colors() > 1:
    #     print("top can be split further")
    # else:
    #     print("top is a single color")

    # if candidate_b.histogram_right.number_of_unique_colors() > 1:
    #     print("bottom can be split further")
    # else:
    #     print("bottom is a single color")
    # return 'top|bottom'
    return 'ok'

class TestImageLayerRepresentation(unittest.TestCase):
    def xtest_10000_a(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        actual = process(image)
        print(actual)
