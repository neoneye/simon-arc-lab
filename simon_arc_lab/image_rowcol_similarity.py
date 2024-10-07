import numpy as np

def image_color_transitions_per_row(image: np.array) -> list[list[int]]:
    """
    Extract the color transitions per row.
    """
    height, width = image.shape
    transitions_list = []
    for y in range(height):
        transition_list = []
        last_color = None
        for x in range(width):
            color = image[y, x]
            if color == last_color:
                continue
            transition_list.append(int(color))
            last_color = color
        transitions_list.append(transition_list)
    return transitions_list

# def image_rowcol_similarity(image0: np.array, image1: np.array):
#     """
#     Measure how many rows/columns are the same in both images.
#     The images doesn't have to be the same size.
#     """

#     return []