from collections import Counter
from typing import Tuple
import numpy as np
from .image_skew import *
from .image_util import *

def extract_bigrams(pixels: list[int], outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract all the bigrams from the list.
    """
    result = []
    for i in range(len(pixels)+1):
        if i >= 1:
            # cast from np.uint8 to int
            color0 = int(pixels[i - 1])
        else:
            color0 = outside_value

        if i < len(pixels):
            # cast from np.uint8 to int
            color1 = int(pixels[i])
        else:
            color1 = outside_value

        result.append((color0, color1))
    return result

def image_bigrams_from_top_to_bottom(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract vertical bigrams.
    """
    result = []
    width = image.shape[1]
    for x in range(width):
        pixels = image[:, x]
        bigrams = extract_bigrams(pixels, outside_value)
        result.extend(bigrams)
    return result

def image_bigrams_from_left_to_right(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract horizontal bigrams.
    """
    return image_bigrams_from_top_to_bottom(np.transpose(image), outside_value)

def image_bigrams_from_topleft_to_bottomright(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract diagonal bigrams.
    """
    image2 = image_skew(image, outside_value, SkewDirection.LEFT)
    bigrams = image_bigrams_from_top_to_bottom(image2, outside_value)
    # Remove bigrams with outside_value's in both columns. These are from the skewed padding area.
    removal_bigram = (outside_value, outside_value)
    result = [bigram for bigram in bigrams if bigram != removal_bigram]
    return result

def image_bigrams_from_topright_to_bottomleft(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract diagonal bigrams.
    """
    return image_bigrams_from_topleft_to_bottomright(image_flipx(image), outside_value)

def sorted_unique_bigrams(bigram_list: list[Tuple[int, int]]) -> list[Tuple[int, int]]:
    # delete bigrams where both values are the same
    bigram_list = [bigram for bigram in bigram_list if bigram[0] != bigram[1]]

    # swap the tuple values, so the smallest value comes first, and the largest comes last
    for i in range(len(bigram_list)):
        bigram_list[i] = (min(bigram_list[i]), max(bigram_list[i]))

    # remove duplicates
    counter = Counter(bigram_list)
    result = list(counter.keys())

    # sort the bigrams by the first element, if there is a tie then sort by the second element
    result = sorted(result, key=lambda x: (x[0], x[1]))

    return result

def image_bigrams_direction_all(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract all bigrams.
    """
    bigram_list = []
    bigram_list.extend(image_bigrams_from_top_to_bottom(image, outside_value))
    bigram_list.extend(image_bigrams_from_left_to_right(image, outside_value))
    bigram_list.extend(image_bigrams_from_topleft_to_bottomright(image, outside_value))
    bigram_list.extend(image_bigrams_from_topright_to_bottomleft(image, outside_value))
    return sorted_unique_bigrams(bigram_list)

def image_bigrams_direction_leftright(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract horizontal bigrams.
    """
    bigram_list = image_bigrams_from_left_to_right(image, outside_value)
    return sorted_unique_bigrams(bigram_list)

def image_bigrams_direction_topbottom(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract vertical bigrams.
    """
    bigram_list = image_bigrams_from_top_to_bottom(image, outside_value)
    return sorted_unique_bigrams(bigram_list)

def image_bigrams_direction_topleftbottomright(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract diagonal bigrams.
    """
    bigram_list = image_bigrams_from_topleft_to_bottomright(image, outside_value)
    return sorted_unique_bigrams(bigram_list)

def image_bigrams_direction_toprightbottomleft(image: np.array, outside_value: int) -> list[Tuple[int, int]]:
    """
    Extract diagonal bigrams.
    """
    bigram_list = image_bigrams_from_topright_to_bottomleft(image, outside_value)
    return sorted_unique_bigrams(bigram_list)

