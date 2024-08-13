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

    # Horizontal - Considers only 2 neighbors, left and right around the center pixel.
    LR2 = 'lr2'

    # Vertical - Considers only 2 neighbors, top and bottom around the center pixel.
    TB2 = 'tb2'

    # Diagonal A - Considers only 2 neighbors, top-left and bottom-right around the center pixel.
    TLBR2 = 'tlbr2'

    # Diagonal B - Considers only 2 neighbors, top-right and bottom-left around the center pixel.
    TRBL2 = 'trbl2'
