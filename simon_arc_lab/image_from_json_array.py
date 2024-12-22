import numpy as np

def image_from_json_array(json_array: list[list[int]], padding: int = 255) -> np.array:
    """
    Convert a list of lists of integers into a numpy array.

    This data usually comes from a LLM response, so the rows may have different lengths.
    """
    if not isinstance(json_array, list):
        raise ValueError(f"Expected json_array to be a list, but got: {type(json_array)}")
    width = 0
    for row in json_array:
        width = max(width, len(row))
    height = len(json_array)
    image = np.full((height, width), padding, dtype=np.uint8)
    for y, row in enumerate(json_array):
        for x, pixel in enumerate(row):
            image[y, x] = pixel
    return image
