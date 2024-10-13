# Connected component labeling/analysis
# 
# https://en.wikipedia.org/wiki/Connected-component_labeling
import numpy as np
from .pixel_connectivity import PixelConnectivity
from .image_fill import *

class ConnectedComponentItem:
    def __init__(self, mask: np.array, mass: int, x: int, y: int):
        self.mask = mask
        self.mass = mass
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, ConnectedComponentItem):
            return (self.mask == other.mask).all() and self.mass == other.mass and self.x == other.x and self.y == other.y
        return False

    def __repr__(self):
        return f"ConnectedComponentItem(mask=..., mass={self.mass}, x={self.x}, y={self.y})"

class ConnectedComponent:
    @staticmethod
    def find_objects_with_ignore_mask_inner(connectivity: PixelConnectivity, image: np.array, ignore_mask: np.array) -> list[ConnectedComponentItem]:
        """
        Identify clusters of connected pixels with an `ignore_mask` of areas to be ignored
        
        Each object is a mask, where it's 1 the object is present, where it's 0 there is no object.
        
        Counts the number of pixels in each of the objects, so that this costly operation can be avoided.
        """
        if ignore_mask.shape != image.shape:
            raise ValueError("The size of the ignore_mask must be the same, but is different")
        
        object_mask_vec = []
        accumulated_mask = ignore_mask.copy()
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Only visit pixels that have not yet been visited
                mask_value = accumulated_mask[y, x]
                if mask_value > 0:
                    # This pixel has already been visited, ignore it
                    continue

                # Flood fill
                color = image[y, x]
                object_mask = ignore_mask.copy()

                image_mask_flood_fill(object_mask, image, x, y, color, connectivity)

                # Clear pixels that are in the original ignore_mask
                object_mask[ignore_mask > 0] = 0

                # Copy the mask into the accumulated mask, so that the pixel doesn't get visited again
                mass = np.count_nonzero(object_mask)
                if mass == 0:
                    continue

                # Count the number of pixels in the mask that are non-zero.
                #
                # Determine the top/left coordinate of where the mask has a non-zero pixel.
                non_zero_coords = np.argwhere(object_mask > 0)
                first_nonzero_pixel_y, first_nonzero_pixel_x = non_zero_coords[0]

                accumulated_mask[object_mask > 0] = 1

                item = ConnectedComponentItem(
                    mask=object_mask,
                    mass=min(mass, 2**16 - 1),
                    x=first_nonzero_pixel_x,
                    y=first_nonzero_pixel_y
                )
                object_mask_vec.append(item)
        
        return object_mask_vec

    @staticmethod
    def find_objects(connectivity: PixelConnectivity, image: np.array) -> list[np.array]:
        """
        Identify clusters of connected pixels
        
        Each object is a mask, where it's 1 the object is present, where it's 0 there is no object.
        """
        if not isinstance(connectivity, PixelConnectivity):
            raise ValueError("connectivity must be a PixelConnectivity enum")
        ignore_mask = np.zeros_like(image)
        return ConnectedComponent.find_objects_with_ignore_mask(connectivity, image, ignore_mask)

    @staticmethod
    def find_objects_with_ignore_mask(connectivity: PixelConnectivity, image: np.array, ignore_mask: np.array) -> list[np.array]:
        """
        Identify clusters of connected pixels with an `ignore_mask` of areas to be ignored
        
        Each object is a mask, where it's 1 the object is present, where it's 0 there is no object.
        """
        if not isinstance(connectivity, PixelConnectivity):
            raise ValueError("connectivity must be a PixelConnectivity enum")
        items = ConnectedComponent.find_objects_with_ignore_mask_inner(connectivity, image, ignore_mask)
        return [item.mask for item in items]
