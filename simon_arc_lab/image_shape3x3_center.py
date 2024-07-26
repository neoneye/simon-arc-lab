import numpy as np

class ImageShape3x3Center:
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 4
    LEFT = 8
    RIGHT = 16
    BOTTOM_LEFT = 32
    BOTTOM = 64
    BOTTOM_RIGHT = 128

    @classmethod
    def apply(cls, image: np.array) -> np.array:
        """
        Identify the shape of the 3x3 neighborhood of each pixel in the image.

        The shape is represented as an 8-bit integer, where each bit corresponds to a neighbor.
        The bit is set to 1 if the neighbor has the same color as the center pixel.

        When the entire mask is zero, it means no neighbors have the same color as the center pixel.

        :param image: image
        return: image with the same size as the input image
        """

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                center_color = image[y, x]
                same_color_mask = 0
                
                # Check all 8 neighbors, in total 8 bits, are they the same color as the center pixel
                current_bit = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if image[ny, nx] == center_color:
                                same_color_mask |= 1 << current_bit
                        current_bit += 1
                
                new_image[y, x] = same_color_mask
        
        return new_image
