from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.image_erosion import *

def image_erosion_multicolor(image: np.ndarray, connectivity: PixelConnectivity) -> np.ndarray:
    component_list = ConnectedComponent.find_objects(connectivity, image)
    accumulated_mask = np.zeros_like(image)
    for component in component_list:
        eroded_mask = image_erosion(component, connectivity)
        accumulated_mask = np.maximum(accumulated_mask, eroded_mask)
    return accumulated_mask
