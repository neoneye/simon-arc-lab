import numpy as np
from .histogram import Histogram

def object_mass(objects: list[np.array]) -> np.array:
    """
    Assign the mass to every pixel of an object.
    
    The pixels for the background is assigned `value 0`.
    
    The pixels for objects with `mass=1` are assigned `value 1`.

    The pixels for objects with `mass=2` are assigned `value 2`.

    Object that have a mass greater than 255 are assigned `value 255`.
    
    There can minimum be 1 object. If zero objects are provided then an exception is raised.
    
    There is no maximum object count.
    
    The objects doesn't have to cover the entire area. The areas not covered by any object is assigned the `value 0`.
    
    Each object is a mask, where it's 1 the object is present, where it's 0 there is no object.
    If the object mask contains values that isn't 0 or 1, then an exception is raised.
    
    The objects are supposed to "not overlap" with each other. If they do overlap then an exception is raised.
    
    All the objects are supposed to have the same `width x height`, otherwise an exception is raised.
    The size of the output image is `width x height`.
    """

    if not objects:
        raise ValueError("Expected minimum 1 object")

    # Determine the size of the result image
    width = objects[0].shape[1]
    height = objects[0].shape[0]

    # Verify that all objects have the same size
    for obj in objects:
        if obj.shape[1] != width or obj.shape[0] != height:
            raise ValueError("Expected all objects to have same size")

    # The size must not be empty
    if width == 0 or height == 0:
        raise ValueError("The size of the objects must be 1x1 or bigger")

    # Enumerate the objects
    result_image = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        histogram = Histogram.create_with_image(obj)
        object_mass = histogram.get_count_for_color(1)

        # Object that are too big to fit inside a uint8 are assigned the value 255
        if object_mass > 255:
            object_mass = 255

        # Draw the object with the object_mass
        for y in range(height):
            for x in range(width):
                object_pixel_value = obj[y, x]
                if object_pixel_value not in {0, 1}:
                    raise ValueError("Invalid mask, the object mask is supposed to be values in the range 0..1")
                if object_pixel_value == 0:
                    continue
                existing_pixel_value = result_image[y, x]
                if existing_pixel_value > 0:
                    raise ValueError("Detected overlap between objects")
                result_image[y, x] = object_mass

    return result_image
