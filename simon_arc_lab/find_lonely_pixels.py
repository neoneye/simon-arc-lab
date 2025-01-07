import numpy as np

def find_lonely_pixels(image: np.array) -> np.array:
    """
    Find the isolated pixels, this often occur in patterns where two colors alternate, such as checkerboards.

    :param image: The image to analyze
    :return: An image where the isolated pixels are marked with 2, strong confidence. 
    Ambiguous lonely pixels are marked with 1, weak confidence.
    Pixels that belong to a bigger object are marked with 0, strong confidence.
    """

    height, width = image.shape

    # Assign 1 to areas where the mass is 1. Lonely pixels.
    # Assign 0 to areas where the mass is +2.
    mask_of_lonely_pixels = np.ones_like(image, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            center_color = image[y, x]
            
            # Check the 4 nearest neighbors
            positions = [(0, -1), (-1, 0), (1, 0), (0, 1)]

            same_color = False
            for dx, dy in positions:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if image[ny, nx] == center_color:
                        same_color = True
                        break
            if same_color:
                mask_of_lonely_pixels[y, x] = 0
    
    # Do a second pass to identify the ambiguous pixels
    # The diagonals are harder to check. 
    # It's ambiguous if it's a lonely pixel or part of a longer solid diagonal line.
    # The ambiguous are marked with 1, since it may be a false positive. Weak confidence.
    # The unambiguous pixels are marked with 2, for indicating stronger confidence.
    result_image = np.zeros_like(image, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if mask_of_lonely_pixels[y, x] == 0:
                continue

            center_color = image[y, x]
            
            # Check the 4 corners
            positions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

            same_color_in_diagonal = False
            for dx, dy in positions:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if mask_of_lonely_pixels[ny, nx] == 0:
                        continue
                    if image[ny, nx] == center_color:
                        same_color_in_diagonal = True
                        break
            value = 1 if same_color_in_diagonal else 2
            result_image[y, x] = value
            
    return result_image
