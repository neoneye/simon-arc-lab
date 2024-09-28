import numpy as np
from .rectangle import Rectangle

def find_bounding_box_ignoring_color(image: np.array, ignore_color: int) -> Rectangle:
    """
    Find the bounding box after trimming one color.
    """
    return find_bounding_box_multiple_ignore_colors(image, [ignore_color])

def find_bounding_box_multiple_ignore_colors(image: np.array, ignore_colors: list[int]) -> Rectangle:
    """
    Find the bounding box after trimming with many colors.
    """
    if image.size == 0:
        return Rectangle.empty()

    height, width = image.shape
    x_max = width - 1
    y_max = height - 1
    found_x0 = x_max
    found_x1 = 0
    found_y0 = y_max
    found_y1 = 0

    ignore_colors_set = set(ignore_colors)
    for y in range(y_max + 1):
        for x in range(x_max + 1):
            pixel_value = image[y, x]
            if pixel_value in ignore_colors_set:
                continue

            # Grow the bounding box
            found_x0 = min(found_x0, x)
            found_x1 = max(found_x1, x)
            found_y0 = min(found_y0, y)
            found_y1 = max(found_y1, y)

    if found_x0 > found_x1 or found_y0 > found_y1:
        return Rectangle.empty()

    # Left position
    if found_x0 < 0 or found_x0 > x_max:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. found_x0: {found_x0}")

    # Top position
    if found_y0 < 0 or found_y0 > y_max:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. found_y0: {found_y0}")

    # Width
    new_width = found_x1 - found_x0 + 1
    if new_width < 1:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. new_width: {new_width}")

    # Height
    new_height = found_y1 - found_y0 + 1
    if new_height < 1:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. new_height: {new_height}")

    return Rectangle(found_x0, found_y0, new_width, new_height)
