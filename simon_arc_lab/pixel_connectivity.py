from enum import Enum

class PixelConnectivity(Enum):
    # Considers only the 4 neighbors around the center pixel, the top/bottom/left/right pixels.
    #
    # Don't visit the corners.
    CONNECTIVITY4 = 4

    # Considers all the 8 neighbors around the center pixel.
    #
    # This can be useful for diagonal flood filling via corners.
    CONNECTIVITY8 = 8

    # Considers only the 4 corners around the center pixel, the top-left/top-right/bottom-left/bottom-right pixels.
    #
    # Don't visit the neighboring pixels.
    CONNECTIVITY4DIAGONAL = 'diagonal'

