import numpy as np

class ImageShape3x3Opposite:
    TOPLEFT_BOTTOMRIGHT = 1
    TOPCENTER_BOTTOMCENTER = 2
    TOPRIGHT_BOTTOMLEFT = 4
    CENTERLEFT_CENTERRIGHT = 8

    @classmethod
    def apply(cls, image: np.array) -> np.array:
        """
        Check where the opposite pixel have same color inside a 3x3 neighborhood.

        There are 4 opposite pixels and is represented as an 4-bit integer, where each bit corresponds to an opposite.
        The bit is set to 1 when two opposite pixels have the same color.

        :param image: image
        return: image with the same size as the input image
        """

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                dx_dy_list = [(1, 1), (0, 1), (1, -1), (1, 0)]
                same_color_mask = 0
                
                # Check all 4 opposites, in total 4 bits, are the opposites the same color
                current_bit = 0
                for dx_dy in dx_dy_list:
                    dx = dx_dy[0]
                    dy = dx_dy[1]
                    ny, nx = y + dy, x + dx
                    my, mx = y - dy, x - dx
                    if 0 <= ny < height and 0 <= nx < width and 0 <= my < height and 0 <= mx < width:
                        if image[ny, nx] == image[my, mx]:
                            same_color_mask |= 1 << current_bit
                    current_bit += 1
                
                new_image[y, x] = same_color_mask
        
        return new_image
