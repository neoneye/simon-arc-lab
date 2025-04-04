"""
RLE compression

This algorithm applies run-length encoding (RLE) to images, reducing sequences of identical pixels to shorter symbolic forms. 
It starts by noting the image's width and height, then processes each line. If an entire line is the same color, it writes that color once. 
Otherwise, it breaks the line into consecutive runs of identical pixels, using letters `a` through `z` to represent runs of lengths 2 to 27, 
followed by the color. Single pixels are just written as their color value. When a line repeats the previous one, it's marked with a comma only. 
This yields a concise, text-based compression.
"""
import numpy as np
from .deserialize import deserialize

def serialize(image):
    """
    Serialize an image to a RLE string, and verifies that it can be deserialized back to the original image.

    :param image: The image to serialize
    :return: The RLE string of the image
    """

    rle_string = serialize_without_verify(image)
    verify_pixels = deserialize(rle_string)
    if not np.array_equal(image, verify_pixels):
        raise Exception("Mismatch between serialize and deserialize of image.")
    return rle_string

def serialize_without_verify(image):
    height, width = image.shape
    s = f"{width} {height} "
    last_line = ""
    for y in range(height):
        if y > 0:
            s += ','
        current_line = rle_serialize_line(image[y, :])
        if current_line != last_line:
            s += current_line
            last_line = current_line

    return s

def rle_serialize_line(line):
    color = line[0]
    is_same_color = np.all(line == color)
    if is_same_color:
        return str(color)
    return rle_serialize_line_inner(line)
    
def rle_serialize_line_inner(line):
    width = len(line)
    current_line = ""
    color = line[0]
    count = 1
    for x in range(1, width):
        new_color = line[x]
        if count < 27 and new_color == color:
            count += 1
            continue
        if count >= 2:
            current_line += chr((count - 2) + ord('a'))
        current_line += str(color)
        color = new_color
        count = 1
    if count >= 2:
        current_line += chr((count - 2) + ord('a'))
        current_line += str(color)
    else:
        current_line += str(color)

    return current_line
