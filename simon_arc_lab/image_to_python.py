import numpy as np
from dataclasses import dataclass
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
    def __init__(self, original_image: np.array, background_color: Optional[int]):
        height, width = original_image.shape
        self.original_image = original_image
        self.original_image_height = height
        self.original_image_width = width

        if background_color is None:
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

@dataclass
class ImageToPythonConfig:
    max_depth = 10
    verbose = False
    verify_bsptree_recreates_image = True
    verify_python_recreates_image = True
    background_color = None

class ImageToPython:
    def __init__(self, image: np.array, config: ImageToPythonConfig):
        self.image = image
        self.config = config
        self.node = None
        self.python_code = None

    def build(self):
        verbose = self.config.verbose
        node = create_bsp_tree(self.image, self.config.max_depth, verbose)
        self.node = node
        if self.config.verify_bsptree_recreates_image:
            recreated_image_from_bsp = node.to_image()
            if np.array_equal(self.image, recreated_image_from_bsp) == False:
                raise Exception("The image recreated from BSP is not equal to the original image.")
        
        v = PythonGeneratorVisitor(self.image, background_color=self.config.background_color)
        node.accept(v)
        python_code = v.get_code()
        self.python_code = python_code
        if verbose:
            print(f"# python code that recreates the image\n{python_code}\n\n")

        if self.config.verify_python_recreates_image:
            # Prepare a separate namespace for exec
            exec_namespace = {}
            
            # Optionally, include necessary imports in the namespace
            exec_namespace['np'] = np
            
            try:
                # Execute the generated Python code within the namespace
                exec(python_code, exec_namespace)
            except Exception as e:
                print("Error executing the generated code:", e)
                bsp_tree = node.tree_to_string("|")
                print("bsp_tree=", bsp_tree)
                raise e
            
            # Retrieve the 'image' variable from the namespace
            recreated_image_from_python = exec_namespace.get('image', None)

            if np.array_equal(self.image, recreated_image_from_python) == False:
                if verbose:
                    print(f"Argh, the python code does not recreates the original image!")
                raise Exception("The image recreated from Python is not equal to the original image.")

        if verbose:
            print(f"Great, the python code recreates the image correctly!")
