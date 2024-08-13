from enum import Enum

class PixelConnectivity(Enum):
    # Considers only the 4 neighbors around the center pixel, the top/bottom/left/right pixels.
    #
    # Don't visit the corners.
    NEAREST4 = 'nearest4'

    # Considers all the 8 neighbors around the center pixel.
    #
    # This can be useful for diagonal flood filling via corners.
    ALL8 = 'all8'

    # Considers only the 4 corners around the center pixel, the top-left/top-right/bottom-left/bottom-right pixels.
    #
    # Don't visit the neighboring pixels.
    CORNER4 = 'corner4'

