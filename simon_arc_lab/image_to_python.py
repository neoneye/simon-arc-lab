import numpy as np
from .bsp_tree import *

class PythonGeneratorVisitor(NodeVisitor):
    """
    Generate Python code that recreates the image from a BSP tree.

    image=np.zeros((10,20),dtype=np.uint8)
    image[y:y+height, x:x+width] = color # fill a rectangle
    image[y, x:x+width] = color # fill multiple columns
    image[:, x:x+width] = color # fill an entire column
    image[y, x] = color # set a single pixel
    """
    def __init__(self, original_image: np.array):
        height, width = original_image.shape
        self.original_image = original_image
        self.original_image_height = height
        self.original_image_width = width

        histogram = Histogram.create_with_image(original_image)
        background_color = histogram.most_popular_color()
        self.background_color = background_color

        self.lines = []
        if background_color is None or background_color == 0:
            self.lines.append(f"image=np.zeros(({height},{width}),dtype=np.uint8)")
        else:
            self.lines.append(f"image=np.full(({height},{width}),{background_color},dtype=np.uint8)")

    def visit_solid_node(self, node: SolidNode):
        x, y, width, height = node.rectangle()
        assert width >= 1 and height >= 1
        assert x >= 0 and y >= 0

        color = node.color
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

        code = f"image[{y_str},{x_str}]={color}"
        self.lines.append(code)

    def visit_image_node(self, node: ImageNode):
        raise Exception("visit_image_node. The fully populated BSP tree should not contain any ImageNode's")

    def visit_split_node(self, node: SplitNode):
        pass

    def get_code(self) -> str:
        return "\n".join(self.lines)
