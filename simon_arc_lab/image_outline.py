import numpy as np

def image_outline_all8(image: np.array) -> np.array:
    """
    Edge detection of objects in the image.

    The mask value is 1 if the pixel is an edge pixel, otherwise 0.

    :param image: image
    return: image with the same size as the input image
    """

    height, width = image.shape
    new_image = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            center_color = image[y, x]
            all_the_same_color = True
            
            # Check all 8 neighbors that they have the same color as the center pixel
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        all_the_same_color = False
                        break
                    if image[ny, nx] != center_color:
                        all_the_same_color = False
                        break
            
            if all_the_same_color:
                value = 0
            else:
                value = 1
            new_image[y, x] = value
    
    return new_image
