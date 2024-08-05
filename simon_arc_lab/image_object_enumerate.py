import numpy as np

def object_enumerate(objects: list[np.array]) -> np.array:
    """
    Assign a unique value to each object.
    
    The pixels for the background is assigned `value 0`.
    
    The pixels for `objects[0]` is assigned `value 1`.
    
    The pixels for `objects[1]` is assigned `value 2`.
    
    The pixels for the `Nth` object is assigned `value N-1`.
    
    There can minimum be 1 object. If zero objects are provided then an exception is raised.
    
    There can maximum be 255 objects. If more objects are provided then an exception is raised.
    
    The objects doesn't have to cover the entire area. The areas not covered by any object is assigned the `value 0`.
    
    Each object is a mask, where it's 1 the object is present, where it's 0 there is no object.
    If the object mask contains values that isn't 0 or 1, then an exception is raised.
    
    The objects are supposed to "not overlap" with each other. If they do overlap then an exception is raised.
    
    All the objects are supposed to have the same `width x height`, otherwise an exception is raised.
    The size of the output image is `width x height`.
    """

    if len(objects) > 255:
        raise ValueError("Expected maximum 255 objects")

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
    for index, obj in enumerate(objects):
        object_id = min(index + 1, 255)

        # Draw the object with the object_id
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
                result_image[y, x] = object_id

    return result_image
