import numpy as np

def image_to_string(image: np.array) -> str:
    height, _ = image.shape
    rows = []
    for y in range(height):
        pixels = image[y, :]
        s = "".join([str(pixel) for pixel in pixels])
        rows.append(s)
    return "\n".join(rows)

def image_from_string(s: str) -> np.array:
    rows = s.split("\n")
    height = len(rows)
    max_width = 0
    for y, row in enumerate(rows):
        max_width = max(max_width, len(row))
    width = max_width
    empty_color = 255
    image = np.full((height, width), empty_color, dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c.isdigit():
                image[y, x] = int(c)
    return image
