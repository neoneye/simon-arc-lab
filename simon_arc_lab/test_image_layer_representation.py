import unittest
from typing import Dict, Tuple, Optional
import numpy as np
# from .image_layer_representation import *
from .histogram import Histogram

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
    def __init__(self, split: int, score: int, image_left: np.array, image_right: np.array, histogram_left: Histogram, histogram_right: Histogram):
        self.split = split
        self.score = score
        self.image_left = image_left
        self.image_right = image_right
        self.histogram_left = histogram_left
        self.histogram_right = histogram_right

    def __str__(self):
        return f"split:{self.split}, score:{self.score}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def find_split(image: np.array) -> Optional['Split2']:
        verbose = True
        height, width = image.shape

        found_score = 0
        found_image_left = None
        found_image_right = None
        found_histogram_left = None
        found_histogram_right = None
        # find optimal left-right split determined by score
        for i in range(width-1):
            split = i + 1
            image_left, image_right = image_split2_left_right(image, split)
            width_left = split
            width_right = width - split
            assert width_left == image_left.shape[1]
            assert width_right == image_right.shape[1]
            histogram_left = Histogram.create_with_image(image_left)
            histogram_right = Histogram.create_with_image(image_right)
            color_diff = histogram_left.unique_colors_set() - histogram_right.unique_colors_set()
            if verbose:
                print(f"split:{split}, left:{histogram_left.number_of_unique_colors()}, right:{histogram_right.number_of_unique_colors()} diff:{color_diff}")
            if len(color_diff) == 0:
                continue
            width_diff = abs(width_left - width_right)
            score = len(color_diff) * (width - width_diff)
            if score <= found_score:
                continue
            if verbose:
                print(f"found better split:{split}, score:{score}")
            found_score = score
            found_image_left = image_left
            found_image_right = image_right
            found_histogram_left = histogram_left
            found_histogram_right = histogram_right
        if found_score == 0:
            if verbose:
                print("no candidates found")
            return None
        if verbose:
            print(f"found_score:{found_score}")
        return Split2(split, found_score, found_image_left, found_image_right, found_histogram_left, found_histogram_right)
    

def process(image: np.array) -> str:
    print("----------- simon is testing ---------")
    height, width = image.shape

    candidate_a = Split2.find_split(image)
    image_transposed = image.transpose()
    candidate_b = Split2.find_split(image_transposed)

    print(f"candidate_a:{candidate_a}")
    print(f"candidate_b:{candidate_b}")
    score_a = 0 if candidate_a is None else candidate_a.score
    score_b = 0 if candidate_b is None else candidate_b.score
    if score_a == 0 and score_b == 0:
        return 'none'
    if score_a > score_b:
        if candidate_a.histogram_left.number_of_unique_colors() > 1:
            print("left can be split further")
        else:
            print("left is a single color")

        if candidate_a.histogram_right.number_of_unique_colors() > 1:
            print("right can be split further")
        else:
            print("right is a single color")

        return 'left|right'

    if candidate_b.histogram_left.number_of_unique_colors() > 1:
        print("top can be split further")
    else:
        print("top is a single color")

    if candidate_b.histogram_right.number_of_unique_colors() > 1:
        print("bottom can be split further")
    else:
        print("bottom is a single color")
    return 'top|bottom'

class TestImageLayerRepresentation(unittest.TestCase):
    def xtest_10000_a(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        actual = process(image)
        print(actual)
