import numpy as np

def _image_rotate45(original: np.array, fill_color: int, is_clockwise: bool) -> np.array:
    """
    Rotates the image by 45 degrees either clockwise or counter-clockwise.

    :param original: The original image as a NumPy array.
    :param fill_color: The fill color to use for empty spaces in the rotated image.
    :param is_clockwise: If True, rotates the image clockwise; otherwise, counter-clockwise.
    :return: The rotated image as a NumPy array.
    """
    # Handle empty image
    if original.size == 0:
        return original.copy()

    # Handle 1x1 image
    if original.shape[0] == 1 and original.shape[1] == 1:
        return original.copy()

    # Compute combined size
    combined_size = original.shape[0] + original.shape[1] - 1
    if combined_size > 255:
        raise ValueError(f"Unable to rotate image. The combined width and height is too large: {combined_size}")

    # Create new image filled with the fill color
    rotated_image = np.full((combined_size, combined_size), fill_color, dtype=original.dtype)

    height, width = original.shape

    # Copy pixels from the original image to the rotated image
    for get_y in range(height):
        for get_x in range(width):
            pixel_value = original[get_y, get_x]
            set_x = get_x + get_y
            set_y = get_x - get_y + (height - 1)

            if 0 <= set_y < combined_size and 0 <= set_x < combined_size:
                rotated_image[set_y, set_x] = pixel_value
            else:
                raise ValueError(f"Integrity error. Unable to set pixel ({set_x}, {set_y}) inside the result image")

    # Flip image if necessary
    if is_clockwise:
        rotated_image = np.transpose(rotated_image)
    else:
        rotated_image = np.flipud(rotated_image)

    return rotated_image

def image_rotate_cw_45(original: np.array, fill_color: int) -> np.array:
    """
    Rotates the image 45 degrees clockwise.

    :param original: The original image as a NumPy array.
    :param fill_color: The fill color to use for empty spaces in the rotated image.
    :return: The rotated image as a NumPy array.
    """
    return _image_rotate45(original, fill_color, is_clockwise=True)

def image_rotate_ccw_45(original: np.array, fill_color: int) -> np.array:
    """
    Rotates the image 45 degrees counter-clockwise.

    :param original: The original image as a NumPy array.
    :param fill_color: The fill color to use for empty spaces in the rotated image.
    :return: The rotated image as a NumPy array.
    """
    return _image_rotate45(original, fill_color, is_clockwise=False)
