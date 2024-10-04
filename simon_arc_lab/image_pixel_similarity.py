import numpy as np

def image_pixel_similarity(image0: np.array, image1: np.array) -> dict:
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
