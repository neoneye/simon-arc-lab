from typing import Dict, Tuple
import numpy as np

def image_pixel_similarity_dict(image0: np.array, image1: np.array) -> Dict[int, Tuple[int, int]]:
    """
    Measure how many pixels are the same in both images.

    return: {color: (count_intersection, count_union)}
    """
    dict = {}
    if (image0.shape != image1.shape):
        return dict
    
    height, width = image0.shape
    for y in range(height):
        for x in range(width):
            color0 = image0[y, x]
            color1 = image1[y, x]
            if color0 == color1:
                colors = [color0]
                same = True
            else:
                colors = [color0, color1]
                same = False
            for color in colors:
                count_intersection, count_union = dict.get(color, (0, 0))
                if same:
                    count_intersection += 1
                count_union += 1
                dict[color] = (count_intersection, count_union)

    return dict

def jaccard_index_from_image_pixel_similarity_dict(dict: dict) -> int:
    """
    Calculate the Jaccard index from the result of the image_pixel_similarity dictionary.

    return: A score between 0 (least similar) and 100 (most similar).
    """

    count_intersection = 0
    count_union = 0
    for color in dict:
        count_intersection += dict[color][0]
        count_union += dict[color][1]
    if count_union == 0:
        return 100
    return 100 * count_intersection // count_union

def image_pixel_similarity_jaccard_index(image0: np.array, image1: np.array) -> int:
    """
    Measure how many pixels are the same in both images.
    Calculate the Jaccard index.

    return: A score between 0 (least similar) and 100 (most similar).
    """
    if (image0.shape != image1.shape):
        return 0
    dict = image_pixel_similarity_dict(image0, image1)
    return jaccard_index_from_image_pixel_similarity_dict(dict)
