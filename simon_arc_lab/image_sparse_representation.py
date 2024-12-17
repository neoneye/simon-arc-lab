import numpy as np
from typing import Tuple, Optional
import re

def image_to_dictionary(image: np.array, include_size: bool, background_color: Optional[int]) -> dict:
    """
    Creates a python dictionary with x, y coordinates as keys and colors as values.
    
    The background_color parameter is optional.
    This is for omitting the most popular color, which is typically the background color.
    But only do so if all the training pairs agree on the same color.
    This can reduce the amount of text outputted.
    
    If include_size is false, then there is no width and height info in the dictionary.
    Returns a string like {(0,0):7,(1,0):7,(2,0):9,(0,1):8,(1,1):7,(2,1):9}
    
    If include_size is true, then it will include the width and height of the image, like this
    {'width':3,'height':2,(0,0):0,(1,0):1,(2,0):2,(0,1):0,(1,1):1,(2,1):2}
    """
    height, width = image.shape
    items = []
    if include_size:
        items.append(f"'width':{width}")
        items.append(f"'height':{height}")
    if background_color is not None:
        items.append(f"'background':{background_color}")
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            if pixel == background_color:
                continue
            items.append(f"({x},{y}):{pixel}")
    return "{" + ",".join(items) + "}"

# Equivalent to the lazy_static regex in Rust
EXTRACT_STRING_VALUE = re.compile(r"'(\w+)'\s*:\s*(\d+)")
EXTRACT_X_Y_COLOR = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+)")

def dictionary_to_image(input_str: str) -> Tuple[np.array, str|None]:
    # Extract width, height, background
    found_width = None
    found_height = None
    found_background = None
    for match in EXTRACT_STRING_VALUE.finditer(input_str):
        key = match.group(1)
        value = int(match.group(2))
        if key == "width":
            found_width = value
        elif key == "height":
            found_height = value
        elif key == "background":
            found_background = value

    if found_width is None:
        raise ValueError("Missing 'width'")
    if found_height is None:
        raise ValueError("Missing 'height'")

    # Create the image, fill_color is found_background or 255 if none
    fill_color = found_background if found_background is not None else 255
    image = np.full((found_height, found_width), fill_color, dtype=np.uint8)

    # Assign pixel values
    count_outside = 0
    for match in EXTRACT_X_Y_COLOR.finditer(input_str):
        x = int(match.group(1))
        y = int(match.group(2))
        color = int(match.group(3))
        # Check if pixel is inside image bounds
        if 0 <= y < found_height and 0 <= x < found_width:
            image[y, x] = color
        else:
            count_outside += 1

    # Count unassigned pixels (those still at 255)
    count_unassigned = np.count_nonzero(image == 255)

    problems = []
    if count_outside > 0:
        problems.append(f"{count_outside} pixels outside")
    if count_unassigned > 0:
        problems.append(f"{count_unassigned} unassigned pixels")

    status = None if len(problems) == 0 else ", ".join(problems)

    return image, status
