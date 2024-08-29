import numpy as np
from .pixel_connectivity import *
from .connected_component import *
from .image_object_mass import *

def image_mass_compare_adjacent_rows(image: np.array, color_same: int, color_top: int, color_bottom: int) -> np.array:
    """
    Compare the length of line segments between adjacent rows.

    :param image: image with colors 0-9.
    :param color_same: when the length is the same.
    :param color_top: when the top row is greater.
    :param color_bottom: when the bottom row is greater.
    :return: image with shape (height-1, width) and with the assigned colors.
    """
    height, width = image.shape
    if width < 1 or height < 2:
        raise ValueError("Expected image size. width => 1 and height >= 2.")

    # Find components of each row
    connectivity = PixelConnectivity.LR2
    component_list = ConnectedComponent.find_objects(connectivity, image)
    if len(component_list) == 0:
        raise Exception("No connected components found.")

    # Measure length of line segments
    mass_image = object_mass(component_list)
    
    output_width = width
    output_height = height - 1

    output_image = np.zeros((output_height, output_width), dtype=np.uint8)
    for y in range(output_height):
        for x in range(output_width):
            # Compare adjacent rows
            a = mass_image[y, x]
            b = mass_image[y + 1, x]
            if a == b:
                value = color_same
            elif a > b:
                value = color_top
            else:
                value = color_bottom
            output_image[y, x] = value

    return output_image

def image_mass_compare_adjacent_columns(image: np.array, color_same: int, color_left: int, color_right: int) -> np.array:
    """
    Compare the length of line segments between adjacent columns.

    :param image: image with colors 0-9.
    :param color_same: when the length is the same.
    :param color_left: when the left column is greater.
    :param color_right: when the right column is greater.
    :return: image with shape (height, width-1) and with the assigned colors.
    """
    height, width = image.shape
    if width < 2 or height < 1:
        raise ValueError("Expected image size. width => 2 and height >= 1.")

    # Find components of each column
    connectivity = PixelConnectivity.TB2 
    component_list = ConnectedComponent.find_objects(connectivity, image)
    if len(component_list) == 0:
        raise Exception("No connected components found.")

    # Measure length of line segments
    mass_image = object_mass(component_list)
    
    output_width = width - 1
    output_height = height

    output_image = np.zeros((output_height, output_width), dtype=np.uint8)
    for y in range(output_height):
        for x in range(output_width):
            # Compare adjacent columns
            a = mass_image[y, x]
            b = mass_image[y, x + 1]
            if a == b:
                value = color_same
            elif a > b:
                value = color_left
            else:
                value = color_right
            output_image[y, x] = value

    return output_image
