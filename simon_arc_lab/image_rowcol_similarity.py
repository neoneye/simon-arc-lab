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

def image_mass_transitions_per_row(image: np.array) -> list[list[int]]:
    """
    Measure the mass of each color span per row.
    """
    height, width = image.shape
    mass_list_list = []
    for y in range(height):
        mass_list = []
        last_color = None
        mass = 0
        for x in range(width):
            color = image[y, x]
            if last_color is None:
                last_color = color
                mass = 1
                continue
            if color == last_color:
                mass += 1
                continue
            mass_list.append(mass)
            last_color = color
            mass = 1
        if mass > 0:
            mass_list.append(mass)
        mass_list_list.append(mass_list)
    return mass_list_list

# def image_rowcol_similarity(image0: np.array, image1: np.array):
#     """
#     Measure how many rows/columns are the same in both images.
#     The images doesn't have to be the same size.
#     """

#     return []