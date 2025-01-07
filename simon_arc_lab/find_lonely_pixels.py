import numpy as np

def find_lonely_pixels(image: np.array) -> np.array:
    """
    Find the isolated pixels, this often occur in patterns where 2 colors alternate, such as checkerboards.

    :param image: The image to analyze
    :return: The mask image where the isolated pixels are marked with 1
    """

    height, width = image.shape
    new_image = np.ones_like(image, dtype=np.uint8)
    
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
                new_image[y, x] = 0
            
    return new_image
