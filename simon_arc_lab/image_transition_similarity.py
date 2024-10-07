# Measure how many rows/columns are the same in both images.
# The images doesn't have to be the same size.
import numpy as np
from enum import Enum

def image_transition_color_per_row(image: np.array) -> list[list[int]]:
    """
    Extract the color transitions per row.
    """
    height, width = image.shape
    color_list_list = []
    for y in range(height):
        color_list = []
        last_color = None
        for x in range(width):
            color = image[y, x]
            if color == last_color:
                continue
            color_list.append(int(color))
            last_color = color
        color_list_list.append(color_list)
    return color_list_list

def intersectionset_of_listlistint(items0: list[list[int]], items1: list[list[int]]) -> set[list[int]]:
    """
    Ignores duplicates in the lists.
    Identify what unique items these lists have in common.
    """
    items_set0 = set(map(tuple, items0))
    items_set1 = set(map(tuple, items1))
    return items_set0 & items_set1

def unionset_of_listlistint(items0: list[list[int]], items1: list[list[int]]) -> set[list[int]]:
    """
    Combine lists, and remove duplicates
    """
    items_set0 = set(map(tuple, items0))
    items_set1 = set(map(tuple, items1))
    return items_set0 | items_set1

def image_transition_similarity_per_row(image0: np.array, image1: np.array) -> tuple[int, int]:
    """
    Measure how many transitions are the same per row.
    The images doesn't have to be the same size.

    If the images are identical then the count_intersection will be the same as count_union.

    Two different images can have the exact same transitions so the count_intersection is equal to count_union.

    return: (count_intersection, count_union)
    """
    value_list_list0 = image_transition_color_per_row(image0)
    value_list_list1 = image_transition_color_per_row(image1)

    union_set = unionset_of_listlistint(value_list_list0, value_list_list1)
    count_union = len(union_set)
    if count_union == 0:
        return (0, 0)

    intersection_set = intersectionset_of_listlistint(value_list_list0, value_list_list1)
    count_intersection = len(intersection_set)
    return (count_intersection, count_union)

def image_transition_similarity_per_column(image0: np.array, image1: np.array) -> tuple[int, int]:
    """
    Measure how many transitions are the same in per column.
    The images doesn't have to be the same size.

    If the images are identical then the count_intersection will be the same as count_union.

    Two different images can have the exact same transitions so the count_intersection is equal to count_union.

    return: (count_intersection, count_union)
    """
    return image_transition_similarity_per_row(np.transpose(image0), np.transpose(image1))

def image_transition_similarity(image0: np.array, image1: np.array) -> tuple[int, int]:
    """
    Measure how many transitions are the same considering rows and columns.
    The images doesn't have to be the same size.

    If the images are identical then the count_intersection will be the same as count_union.

    Two different images can have the exact same transitions so the count_intersection is equal to count_union.

    return: (count_intersection, count_union)
    """
    count_intersection_row, count_union_row = image_transition_similarity_per_row(image0, image1)
    count_intersection_column, count_union_column = image_transition_similarity_per_column(image0, image1)
    return (count_intersection_row + count_intersection_column, count_union_row + count_union_column)
