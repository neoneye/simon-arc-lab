import numpy as np
from .rectangle import Rectangle

def outer_bounding_box_after_trim_with_color(image_array: np.array, color_to_be_trimmed: int) -> Rectangle:
    """
    Find the bounding box after trimming.
    """
    if image_array.size == 0:
        return Rectangle.empty()

    height, width = image_array.shape
    x_max = width - 1
    y_max = height - 1
    found_x0 = x_max
    found_x1 = 0
    found_y0 = y_max
    found_y1 = 0

    for y in range(y_max + 1):
        for x in range(x_max + 1):
            pixel_value = image_array[y, x]
            if pixel_value == color_to_be_trimmed:
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
    x = found_x0

    # Top position
    if found_y0 < 0 or found_y0 > y_max:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. found_y0: {found_y0}")
    y = found_y0

    # Width
    new_width_i32 = found_x1 - found_x0 + 1
    if new_width_i32 < 1:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. new_width_i32: {new_width_i32}")
    width = new_width_i32

    # Height
    new_height_i32 = found_y1 - found_y0 + 1
    if new_height_i32 < 1:
        raise ValueError(f"Integrity error. Bounding box coordinates are messed up. new_height_i32: {new_height_i32}")
    height = new_height_i32

    return Rectangle(x, y, width, height)
