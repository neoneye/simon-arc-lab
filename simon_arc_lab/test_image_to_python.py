import unittest
import numpy as np
from .image_to_python import *
from .bsp_tree import *

def process(image: np.array, max_depth: int, verbose: bool=False) -> str:
    node = create_bsp_tree(image, max_depth, verbose)
    recreated_image_from_bsp = node.to_image()
    if np.array_equal(image, recreated_image_from_bsp) == False:
        raise Exception("The image recreated from BSP is not equal to the original image.")
    
    v = PythonGeneratorVisitor(image)
    node.accept(v)
    python_code = v.get_code()
    if verbose:
        print(f"# python code that recreates the image\n{python_code}\n\n")

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

    if np.array_equal(image, recreated_image_from_python) == False:
        if verbose:
            print(f"Argh, the python code does not recreates the original image!")
        raise Exception("The image recreated from Python is not equal to the original image.")

    if verbose:
        print(f"Great, the python code recreates the image correctly!")
    return python_code

class TestImageToPython(unittest.TestCase):
    def test_10000_solid_color(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        process(image, 1)

    def test_11000_direction_topbottom(self):
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        process(image, 2)

    def test_12000_direction_leftright(self):
        image = np.array([
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1]], dtype=np.uint8)
        process(image, 3)

    def test_13000_direction_leftright_and_direction_topbottom(self):
        image = np.array([
            [5, 5, 3, 1],
            [7, 7, 7, 7],
            [5, 5, 3, 1],
            [5, 5, 3, 1],
            [5, 5, 3, 1]], dtype=np.uint8)
        process(image, 4)

    def test_20000_small_image_with_one_pixel_in_the_center(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        image[1, 1] = 42
        process(image, 4)

    def test_20001_big_image_with_one_pixel_in_the_center(self):
        image = np.zeros((99, 51), dtype=np.uint8)
        image[50, 25] = 1
        process(image, 8)

    def test_20002_big_image_with_2x2_pixels_in_the_center(self):
        image = np.zeros((99, 51), dtype=np.uint8)
        image[50:52, 25:27] = 1
        process(image, 10)
