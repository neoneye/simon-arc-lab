import numpy as np
from typing import Optional
from .histogram import Histogram

class PythonImageBuilder:
    """
    Generate Python code with rectangles

    Example:
    output0=np.zeros((10,10),dtype=np.uint8)
    output0[0,2]=1
    output0[0:2,6]=1
    output0[2:5,1]=2
    output0[2:5,9]=2
    output0[5,:]=5
    output0[6:9,5]=2
    output0[8:10,1]=1
    output0[8:10,8]=1

    It tries to use the shortest code possible to fill the rectangle.
    image=np.zeros((10,20),dtype=np.uint8)
    image[y:y+height, x:x+width] = color # fill a rectangle
    image[y, x:x+width] = color # fill multiple columns
    image[:, x:x+width] = color # fill an entire column
    image[y, x] = color # set a single pixel
    """
    def __init__(self, original_image: np.array, background_color: Optional[int], name: Optional[str]):
        height, width = original_image.shape
        self.original_image = original_image
        self.original_image_height = height
        self.original_image_width = width

        if background_color is None:
            histogram = Histogram.create_with_image(original_image)
            background_color = histogram.most_popular_color()
        self.background_color = background_color

        if name is None:
            name = "image"
        self.name = name

        self.lines = []
        if background_color is None or background_color == 0:
            self.lines.append(f"{name}=np.zeros(({height},{width}),dtype=np.uint8)")
        else:
            self.lines.append(f"{name}=np.full(({height},{width}),{background_color},dtype=np.uint8)")

    def rectangle(self, x: int, y: int, width: int, height: int, color: int):
        assert width >= 1 and height >= 1
        assert x >= 0 and y >= 0

        if color == self.background_color:
            return
        
        if width == 1:
            x_str = str(x)
        else:
            x_str = f"{x}:{x+width}"
        if width == self.original_image_width and x == 0:
            x_str = ":"

        if height == 1:
            y_str = str(y)
        else:
            y_str = f"{y}:{y+height}"
        if height == self.original_image_height and y == 0:
            y_str = ":"

        code = f"{self.name}[{y_str},{x_str}]={color}"
        self.lines.append(code)

    def python_code(self) -> str:
        return "\n".join(self.lines)
