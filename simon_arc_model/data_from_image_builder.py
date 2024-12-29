from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_shape3x3_opposite import ImageShape3x3Opposite
from simon_arc_lab.image_shape3x3_center import ImageShape3x3Center
from simon_arc_lab.image_shape3x3_histogram import *
from simon_arc_lab.image_count3x3 import *
from simon_arc_lab.image_erosion_multicolor import image_erosion_multicolor
from simon_arc_lab.image_raytrace_probecolor import *
from simon_arc_lab.image_outline import *
from simon_arc_lab.image_gravity_draw import *
from simon_arc_lab.image_skew import *
from simon_arc_lab.image_mass_compare import *
from simon_arc_lab.image_rotate45 import *
from simon_arc_lab.pixel_connectivity import PixelConnectivity
from simon_arc_lab.connected_component import *
from simon_arc_lab.find_bounding_box import *
from simon_arc_lab.shape import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.show_prediction_result import show_prediction_result
from enum import Enum
import numpy as np

class Shape3x3Operation(Enum):
    NUMBER_OF_UNIQUE_COLORS_ALL9 = 'number_of_unique_colors_all9'
    NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER = 'number_of_unique_colors_around_center'
    NUMBER_OF_UNIQUE_COLORS_IN_CORNERS = 'number_of_unique_colors_in_corners'
    NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4 = 'number_of_unique_colors_in_diamond4'
    NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5 = 'number_of_unique_colors_in_diamond5'

class DataFromImageBuilder:
    def __init__(self, image: np.array):
        self.image = image
        self.height, self.width = image.shape
        self.pixel_count = self.width * self.height
        self.outside_color = 10
        self.data = {}
        self.cache_connected_component_item_list = {}
        self.cache_object_ids = {}
    
    def make_pair_id(self, pair_id: int):
        self.data['pair_id'] = [pair_id] * self.pixel_count

    def make_is_earlier_prediction(self, is_earlier_prediction: bool):
        value = 0 if is_earlier_prediction else 1
        self.data['is_earlier_prediction'] = [value] * self.pixel_count

    def make_center_pixel(self):
        self.data['center_pixel'] = self.image.flatten().tolist()

    def make_color_popularity(self, lookaround_size: int):
        full_histogram = Histogram.create_with_image(self.image)
        most_popular_color_set = set(full_histogram.most_popular_color_list())
        least_popular_color_set = set(full_histogram.least_popular_color_list())

        k = lookaround_size
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                values_most_popular = []
                values_least_popular = []
                values_medium_popular = []
                for y in range(self.height):
                    for x in range(self.width):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values_most_popular.append(0)
                            values_least_popular.append(0)
                            values_medium_popular.append(0)
                        else:
                            color = self.image[yy, xx]
                            is_most_popular = color in most_popular_color_set
                            values_most_popular.append(int(is_most_popular))
                            is_least_popular = color in least_popular_color_set
                            values_least_popular.append(int(is_least_popular))
                            is_medium_popular = is_most_popular == False and is_least_popular == False
                            values_medium_popular.append(int(is_medium_popular))
                self.data[f'color_popularity_is_most_popular_x{rx}_y{ry}'] = values_most_popular
                self.data[f'color_popularity_is_least_popular_x{rx}_y{ry}'] = values_least_popular
                self.data[f'color_popularity_is_medium_popular_x{rx}_y{ry}'] = values_medium_popular

    def make_position_xy(self):
        values_x = []
        values_y = []
        values_x_rev = []
        values_y_rev = []
        for y in range(self.height):
            for x in range(self.width):
                x_rev = self.width - x - 1
                y_rev = self.height - y - 1
                values_x.append(x)
                values_y.append(y)
                values_x_rev.append(x_rev)
                values_y_rev.append(y_rev)
        self.data['position_x'] = values_x
        self.data['position_y'] = values_y
        self.data['position_x_rev'] = values_x_rev
        self.data['position_y_rev'] = values_y_rev

    def make_position_xy_lookaround(self, lookaround_size: int):
        for i in range(lookaround_size):
            values_x_plus = []
            values_x_minus = []
            values_y_plus = []
            values_y_minus = []
            values_x_rev_plus = []
            values_x_rev_minus = []
            values_y_rev_plus = []
            values_y_rev_minus = []
            for y in range(self.height):
                for x in range(self.width):
                    x_rev = self.width - x - 1
                    y_rev = self.height - y - 1
                    j = i + 1
                    values_x_plus.append(x + j)
                    values_x_minus.append(x - j)
                    values_y_plus.append(y + j)
                    values_y_minus.append(y - j)
                    values_x_rev_plus.append(x_rev + j)
                    values_x_rev_minus.append(x_rev - j)
                    values_y_rev_plus.append(y_rev + j)
                    values_y_rev_minus.append(y_rev - j)
            self.data[f'position_x_plus_{i}'] = values_x_plus
            self.data[f'position_x_minus_{i}'] = values_x_minus
            self.data[f'position_y_plus_{i}'] = values_y_plus
            self.data[f'position_y_minus_{i}'] = values_y_minus
            self.data[f'position_x_rev_plus_{i}'] = values_x_rev_plus
            self.data[f'position_x_rev_minus_{i}'] = values_x_rev_minus
            self.data[f'position_y_rev_plus_{i}'] = values_y_rev_plus
            self.data[f'position_y_rev_minus_{i}'] = values_y_rev_minus

    def make_any_edge(self):
        values = []
        for y in range(self.height):
            for x in range(self.width):
                is_edge = x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1
                values.append(int(is_edge))
        self.data['any_edge'] = values

    def make_any_corner(self):
        values = []
        for y in range(self.height):
            for x in range(self.width):
                x_rev = self.width - x - 1
                y_rev = self.height - y - 1
                is_corner = (x == 0 and y == 0) or (x == 0 and y_rev == 0) or (x_rev == 0 and y == 0) or (x_rev == 0 and y_rev == 0)
                values.append(int(is_corner))
        self.data['any_corner'] = values
    
    def make_diagonal_distance(self):
        steps = [1, 3, 7]
        for step in steps:
            values_x_plus_y = []
            values_x_rev_plus_y = []
            values_x_plus_y_rev = []
            values_x_rev_plus_y_rev = []
            for y in range(self.height):
                for x in range(self.width):
                    x_rev = self.width - x - 1
                    y_rev = self.height - y - 1

                    values_x_plus_y.append(int((x + y) & step > 0))
                    values_x_rev_plus_y.append(int((x_rev + y) & step > 0))
                    values_x_plus_y_rev.append(int((x + y_rev) & step > 0))
                    values_x_rev_plus_y_rev.append(int((x_rev + y_rev) & step > 0))
            self.data[f'diagonal_distance_step{step}_x_plus_y'] = values_x_plus_y
            self.data[f'diagonal_distance_step{step}_x_rev_plus_y'] = values_x_rev_plus_y
            self.data[f'diagonal_distance_step{step}_x_plus_y_rev'] = values_x_plus_y_rev
            self.data[f'diagonal_distance_step{step}_x_rev_plus_y_rev'] = values_x_rev_plus_y_rev

    def make_corner(self):
        values_topleft = []
        values_topright = []
        values_bottomleft = []
        values_bottomright = []
        for y in range(self.height):
            for x in range(self.width):
                x_rev = self.width - x - 1
                y_rev = self.height - y - 1
                corner_topleft = x == 0 and y == 0
                corner_topright = x_rev == 0 and y == 0
                corner_bottomleft = x == 0 and y_rev == 0
                corner_bottomright = x_rev == 0 and y_rev == 0
                values_topleft.append(int(corner_topleft))
                values_topright.append(int(corner_topright))
                values_bottomleft.append(int(corner_bottomleft))
                values_bottomright.append(int(corner_bottomright))
        self.data['corner_topleft'] = values_topleft
        self.data['corner_topright'] = values_topright
        self.data['corner_bottomleft'] = values_bottomleft
        self.data['corner_bottomright'] = values_bottomright

    def make_center_xy(self):
        values_center_x = []
        values_center_y = []
        for y in range(self.height):
            for x in range(self.width):
                x_rev = self.width - x - 1
                y_rev = self.height - y - 1
                is_center_column = abs(x - x_rev) < 2
                is_center_row = abs(y - y_rev) < 2
                values_center_x.append(int(is_center_column))
                values_center_y.append(int(is_center_row))
        self.data['center_x'] = values_center_x
        self.data['center_y'] = values_center_y

    def make_image_pixel_color(self, lookaround_size: int, suppress_center: bool):
        k = lookaround_size
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                if suppress_center and rx == k and ry == k:
                    continue
                values = []
                for y in range(self.height):
                    for x in range(self.width):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values.append(self.outside_color)
                        else:
                            values.append(self.image[yy, xx])
                self.data[f'lookaround_image_pixel_x{rx}_y{ry}'] = values
    
    def components(self, pixel_connectivity: PixelConnectivity) -> list[ConnectedComponentItem]:
        connected_component_item_list = self.cache_connected_component_item_list.get(pixel_connectivity)
        if connected_component_item_list is not None:
            # print(f'Using cached connected_component_item_list for pixel_connectivity={pixel_connectivity}')
            return connected_component_item_list
        # print(f'Computing connected_component_item_list for pixel_connectivity={pixel_connectivity}')
        ignore_mask = np.zeros_like(self.image)
        connected_component_item_list = ConnectedComponent.find_objects_with_ignore_mask_inner(pixel_connectivity, self.image, ignore_mask)
        self.cache_connected_component_item_list[pixel_connectivity] = connected_component_item_list
        return connected_component_item_list
    
    def make_shape(self, pixel_connectivity: PixelConnectivity):
        components = self.components(pixel_connectivity)

        shape_catalog_indexes = np.zeros((self.height, self.width), dtype=np.uint32)
        shape_scale_x = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_scale_y = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_rotate_cw = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_rotate_ccw = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_rotate_180 = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_flip_x = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_flip_y = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_flip_a = np.zeros((self.height, self.width), dtype=np.uint8)
        shape_transformation_flip_b = np.zeros((self.height, self.width), dtype=np.uint8)
        for component in components:
            shape = image_find_shape(component.mask, verbose=False)
            if shape is None:
                continue

            if isinstance(shape, SolidRectangleShape):
                rect = shape.rectangle
                for rel_y in range(rect.height):
                    for rel_x in range(rect.width):
                        x = rect.x + rel_x
                        y = rect.y + rel_y
                        shape_catalog_indexes[y, x] = 99
                        shape_scale_x[y, x] = rect.width
                        shape_scale_y[y, x] = rect.height

                        if rect.width == rect.height:
                            shape_transformation_rotate_cw[y, x] = 1
                            shape_transformation_rotate_ccw[y, x] = 1
                            shape_transformation_rotate_180[y, x] = 1
                            shape_transformation_flip_x[y, x] = 1
                            shape_transformation_flip_y[y, x] = 1
                            shape_transformation_flip_a[y, x] = 1
                            shape_transformation_flip_b[y, x] = 1
                        else:
                            shape_transformation_rotate_180[y, x] = 1
                            shape_transformation_flip_x[y, x] = 1
                            shape_transformation_flip_y[y, x] = 1

            if isinstance(shape, SimpleShape):
                rect = shape.rectangle
                for rel_y in range(rect.height):
                    for rel_x in range(rect.width):
                        x = rect.x + rel_x
                        y = rect.y + rel_y
                        mask_value = component.mask[y, x]
                        if mask_value != 1:
                            continue
                        shape_catalog_indexes[y, x] = shape.shape_catalog_index + 100
                        if shape.scale_x is not None:
                            shape_scale_x[y, x] = shape.scale_x
                        if shape.scale_y is not None:
                            shape_scale_y[y, x] = shape.scale_y
                        if Transformation.ROTATE_CW in shape.transformation_set:
                            shape_transformation_rotate_cw[y, x] = 1
                        if Transformation.ROTATE_CCW in shape.transformation_set:
                            shape_transformation_rotate_ccw[y, x] = 1
                        if Transformation.ROTATE_180 in shape.transformation_set:
                            shape_transformation_rotate_180[y, x] = 1
                        if Transformation.FLIP_X in shape.transformation_set:
                            shape_transformation_flip_x[y, x] = 1
                        if Transformation.FLIP_Y in shape.transformation_set:
                            shape_transformation_flip_y[y, x] = 1
                        if Transformation.FLIP_A in shape.transformation_set:
                            shape_transformation_flip_a[y, x] = 1
                        if Transformation.FLIP_B in shape.transformation_set:
                            shape_transformation_flip_b[y, x] = 1

        self.data[f'shape_catalog_indexes'] = shape_catalog_indexes.flatten().tolist()
        self.data[f'shape_scale_x'] = shape_scale_x.flatten().tolist()
        self.data[f'shape_scale_y'] = shape_scale_y.flatten().tolist()
        self.data[f'shape_transformation_rotate_cw'] = shape_transformation_rotate_cw.flatten().tolist()
        self.data[f'shape_transformation_rotate_ccw'] = shape_transformation_rotate_ccw.flatten().tolist()
        self.data[f'shape_transformation_rotate_180'] = shape_transformation_rotate_180.flatten().tolist()
        self.data[f'shape_transformation_flip_x'] = shape_transformation_flip_x.flatten().tolist()
        self.data[f'shape_transformation_flip_y'] = shape_transformation_flip_y.flatten().tolist()
        self.data[f'shape_transformation_flip_a'] = shape_transformation_flip_a.flatten().tolist()
        self.data[f'shape_transformation_flip_b'] = shape_transformation_flip_b.flatten().tolist()

    def object_ids(self, pixel_connectivity: PixelConnectivity, object_id_start: int) -> np.array:
        cache_key = (pixel_connectivity, object_id_start)
        object_ids_image = self.cache_object_ids.get(cache_key)
        if object_ids_image is not None:
            return object_ids_image
        
        components = self.components(pixel_connectivity)
            
        object_ids = np.zeros((self.height, self.width), dtype=np.uint32)
        for component_index, component in enumerate(components):
            object_id = object_id_start + component_index
            for y in range(self.height):
                for x in range(self.width):
                    mask_value = component.mask[y, x]
                    if mask_value == 1:
                        object_ids[y, x] = object_id
        self.cache_object_ids[cache_key] = object_ids
        return object_ids

    def make_object_shape(self, pixel_connectivity: PixelConnectivity, lookaround_size_shape: int):
        components = self.components(pixel_connectivity)
        object_shape_image = np.zeros((self.height, self.width), dtype=np.uint32)
        for component in components:
            rect = find_bounding_box_ignoring_color(component.mask, 0)
            rect_mass = rect.width * rect.height
            value = 0 # non-solid object
            if component.mass == rect_mass:
                if rect.width > rect.height:
                    value = 1 # solid rectangle, landscape 
                elif rect.width < rect.height:
                    value = 2 # solid rectangle, portrait
                else:
                    value = 3 # solid rectangle, square
            for y in range(self.height):
                for x in range(self.width):
                    mask_value = component.mask[y, x]
                    if mask_value == 1:
                        object_shape_image[y, x] = value

        k = lookaround_size_shape
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                xx = x + rx - k
                yy = y + ry - k
                values = []
                for y in range(self.height):
                    for x in range(self.width):
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values.append(0)
                        else:
                            values.append(object_shape_image[yy, xx])
                self.data[f'component_{pixel_connectivity}_object_shape_x{rx}_y{ry}'] = values

    def make_object_id(self, pixel_connectivity: PixelConnectivity, object_id_start: int, lookaround_size: int):
        object_ids = self.object_ids(pixel_connectivity, object_id_start)
        k = lookaround_size
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                values = []
                for y in range(self.height):
                    for x in range(self.width):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values.append(0)
                        else:
                            values.append(object_ids[yy, xx])
                self.data[f'component_{pixel_connectivity}_object_ids_x{rx}_y{ry}'] = values

    def make_object_mass(self, pixel_connectivity: PixelConnectivity, lookaround_size: int):
        components = self.components(pixel_connectivity)
        object_masses = np.zeros((self.height, self.width), dtype=np.uint32)
        for component in components:
            for y in range(self.height):
                for x in range(self.width):
                    mask_value = component.mask[y, x]
                    if mask_value == 1:
                        object_masses[y, x] = component.mass

        k = lookaround_size
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                values = []
                for y in range(self.height):
                    for x in range(self.width):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values.append(0)
                        else:
                            values.append(object_masses[yy, xx])
                self.data[f'component_{pixel_connectivity}_object_mass_x{rx}_y{ry}'] = values

    def make_distance_inside_object(self, pixel_connectivity: PixelConnectivity):
        components = self.components(pixel_connectivity)
        shape = self.image.shape
        object_distance_topleft = np.zeros(shape, dtype=np.uint32)
        object_distance_topright = np.zeros(shape, dtype=np.uint32)
        object_distance_bottomleft = np.zeros(shape, dtype=np.uint32)
        object_distance_bottomright = np.zeros(shape, dtype=np.uint32)
        object_distance_top = np.zeros(shape, dtype=np.uint32)
        object_distance_bottom = np.zeros(shape, dtype=np.uint32)
        object_distance_left = np.zeros(shape, dtype=np.uint32)
        object_distance_right = np.zeros(shape, dtype=np.uint32)
        for component in components:
            rect = find_bounding_box_ignoring_color(component.mask, 0)
            for y in range(rect.height):
                for x in range(rect.width):
                    mask_value = component.mask[y + rect.y, x + rect.x]
                    if mask_value == 1:
                        object_distance_topleft[rect.y + y, rect.x + x] = x + y
                        object_distance_topright[rect.y + y, (rect.x + rect.width - 1) - x] = rect.width - 1 - x + y
                        object_distance_bottomleft[(rect.y + rect.height - 1) - y, rect.x + x] = x + rect.height - 1 - y
                        object_distance_bottomright[(rect.y + rect.height - 1) - y, (rect.x + rect.width - 1) - x] = rect.width - 1 - x + rect.height - 1 - y
                        object_distance_top[rect.y + y, rect.x + x] = y
                        object_distance_bottom[rect.y + y, rect.x + x] = rect.height - 1 - y
                        object_distance_left[rect.y + y, rect.x + x] = x
                        object_distance_right[rect.y + y, rect.x + x] = rect.width - 1 - x
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_topleft'] = object_distance_topleft.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_topright'] = object_distance_topright.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_bottomleft'] = object_distance_bottomleft.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_bottomright'] = object_distance_bottomright.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_top'] = object_distance_top.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_bottom'] = object_distance_bottom.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_left'] = object_distance_left.flatten().tolist()
        self.data[f'connectivity{pixel_connectivity}_distance_inside_object_right'] = object_distance_right.flatten().tolist()

    def make_shape3x3_opposite(self, lookaround_size: int):
        image_shape3x3_opposite = ImageShape3x3Opposite.apply(self.image)
        k = lookaround_size
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                values = []
                for y in range(self.height):
                    for x in range(self.width):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values.append(0)
                        else:
                            values.append(image_shape3x3_opposite[yy, xx])
                self.data[f'shape3x3_opposite_x{rx}_y{ry}'] = values

        for i in range(3):
            values = []
            for y in range(self.height):
                for x in range(self.width):
                    values.append((image_shape3x3_opposite[y, x] >> i) & 1)
            self.data[f'shape3x3_opposite_bit{i}'] = values

    def make_shape3x3_center(self, lookaround_size: int):
        image_shape3x3_center = ImageShape3x3Center.apply(self.image)
        k = lookaround_size
        n = k * 2 + 1
        for ry in range(n):
            for rx in range(n):
                values = []
                for y in range(self.height):
                    for x in range(self.width):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                            values.append(0)
                        else:
                            values.append(image_shape3x3_center[yy, xx])
                self.data[f'shape3x3_center_x{rx}_y{ry}'] = values

        for i in range(8):
            values = []
            for y in range(self.height):
                for x in range(self.width):
                    values.append((image_shape3x3_center[y, x] >> i) & 1)
            self.data[f'shape3x3_center_bit{i}'] = values

    def make_probe_color_for_all_directions(self):
        outside_color = 10
        ray_directions = [
            ImageRaytraceProbeColorDirection.TOP,
            ImageRaytraceProbeColorDirection.BOTTOM,
            ImageRaytraceProbeColorDirection.LEFT,
            ImageRaytraceProbeColorDirection.RIGHT,
            ImageRaytraceProbeColorDirection.TOPLEFT,
            ImageRaytraceProbeColorDirection.TOPRIGHT,
            ImageRaytraceProbeColorDirection.BOTTOMLEFT,
            ImageRaytraceProbeColorDirection.BOTTOMRIGHT,
        ]
        for direction in ray_directions:
            image_ray = image_raytrace_probecolor_direction(self.image, outside_color, direction)
            self.data[f'probe_color_direction{direction}'] = image_ray.flatten().tolist()

    def make_probe_objectid_for_all_directions(self, object_ids: np.array, pixel_connectivity: PixelConnectivity):
        outside_object_id = 0
        ray_directions = [
            ImageRaytraceProbeColorDirection.TOP,
            ImageRaytraceProbeColorDirection.BOTTOM,
            ImageRaytraceProbeColorDirection.LEFT,
            ImageRaytraceProbeColorDirection.RIGHT,
            ImageRaytraceProbeColorDirection.TOPLEFT,
            ImageRaytraceProbeColorDirection.TOPRIGHT,
            ImageRaytraceProbeColorDirection.BOTTOMLEFT,
            ImageRaytraceProbeColorDirection.BOTTOMRIGHT,
        ]
        for direction in ray_directions:
            image_ray = image_raytrace_probecolor_direction(object_ids, outside_object_id, direction)
            self.data[f'probe_objectid_direction{direction}_connectivity{pixel_connectivity}'] = image_ray.flatten().tolist()

    def make_outline_all8(self):
        the_image_outline_all8 = image_outline_all8(self.image)
        values = []
        for y in range(self.height):
            for x in range(self.width):
                is_outline = the_image_outline_all8[y, x]
                if is_outline == 1:
                    values.append(100)
                else:
                    values.append(-100)
        self.data['image_outline_all8'] = values

    def make_count_same_color_as_center(self, lookaround_size: int):
        n = lookaround_size
        for dy in range(-n, n * 2):
            for dx in range(-n, n * 2):
                if dx == 0 and dy == 0:
                    continue
                values = []
                image_same = count_same_color_as_center_with_one_neighbor_nowrap(self.image, dx, dy)
                for y in range(self.height):
                    for x in range(self.width):
                        is_same = image_same[y, x] == 1
                        if is_same:
                            values.append(100)
                        else:
                            values.append(-100)
                self.data[f'count_same_color_as_center_with_one_neighbor_nowrap_dx{dx}_dy{dy}'] = values

    def make_count_neighbors_with_same_color(self):
        image_count = count_neighbors_with_same_color_nowrap(self.image)
        self.data['count_neighbors_with_same_color'] = image_count.flatten().tolist()

    def assign_data_with_unique_colors(self, histogram_per_pixel: list[Histogram], data_name: str):
        values_color_is_present = []
        for _ in range(10):
            values_color_is_present.append([])

        values_count = []
        for y in range(self.height):
            for x in range(self.width):
                histogram = histogram_per_pixel[y * self.width + x]
                unique_color_set = histogram.unique_colors_set()
                number_of_unique_colors = len(unique_color_set)
                values_count.append(number_of_unique_colors)

                for color in range(10):
                    is_present = color in unique_color_set
                    values_color_is_present[color].append(int(is_present))

        self.data[f'unique_color_count_{data_name}'] = values_count
        for color in range(10):
            self.data[f'unique_color_present_{data_name}_color{color}'] = values_color_is_present[color]

    def assign_data_with_count_of_each_color(self, histogram_per_pixel: list[Histogram], enable_count_with_minus1: bool, data_name: str):
        for color in range(10):
            values_count = []
            values_count_with_minus1 = []
            for y in range(self.height):
                for x in range(self.width):
                    histogram = histogram_per_pixel[y * self.width + x]

                    count = histogram.get_count_for_color(color)
                    values_count.append(count)

                    if count > 0:
                        values_count_with_minus1.append(color)
                    else:
                        values_count_with_minus1.append(-1)
            self.data[f'histogram_{data_name}_color{color}_count'] = values_count

            if enable_count_with_minus1:
                self.data[f'histogram_{data_name}_color{color}_count_with_minus1'] = values_count_with_minus1

    def make_histogram_row(self, enable_count_with_minus1: bool):
        """
        A histogram per row
        """
        row_histograms = []
        for y in range(self.height):
            row = self.image[y, :]
            # convert to 2d image
            row_image = np.expand_dims(row, axis=0)
            histogram = Histogram.create_with_image(row_image)
            row_histograms.append(histogram)

        row_histogram_per_pixel = []
        for y in range(self.height):
            for x in range(self.width):
                row_histogram_per_pixel.append(row_histograms[y])
        self.assign_data_with_unique_colors(row_histogram_per_pixel, 'row')
        self.assign_data_with_count_of_each_color(row_histogram_per_pixel, enable_count_with_minus1, 'row')

    def make_histogram_column(self, enable_count_with_minus1: bool):
        """
        A histogram per column
        """
        column_histogram = []
        for x in range(self.width):
            column = self.image[:, x]
            # convert to 2d image
            column_image = np.expand_dims(column, axis=0)
            histogram = Histogram.create_with_image(column_image)
            column_histogram.append(histogram)

        column_histogram_per_pixel = []
        for y in range(self.height):
            for x in range(self.width):
                column_histogram_per_pixel.append(column_histogram[x])
        self.assign_data_with_unique_colors(column_histogram_per_pixel, 'column')
        self.assign_data_with_count_of_each_color(column_histogram_per_pixel, enable_count_with_minus1, 'column')

    def make_histogram_tlbr(self, enable_count_with_minus1: bool):
        """
        A histogram on the diagonal from top-left to bottom-right
        """
        outside_color = 10
        tlbr_histograms = []
        skewed_image_down = image_skew(self.image, outside_color, SkewDirection.DOWN)
        for y in range(skewed_image_down.shape[0]):
            row = skewed_image_down[y, :]
            # convert to 2d image
            row_image = np.expand_dims(row, axis=0)
            histogram = Histogram.create_with_image(row_image)
            histogram.remove_color(outside_color)
            # print(f'y={y} histogram={histogram.pretty()}')
            tlbr_histograms.append(histogram)
        # show_prediction_result(self.image, skewed_image_down, None)

        tlbr_histogram_per_pixel = []
        for y in range(self.height):
            for x in range(self.width):
                tlbr_histogram_per_pixel.append(tlbr_histograms[x + y])
        self.assign_data_with_unique_colors(tlbr_histogram_per_pixel, 'tlbr')
        self.assign_data_with_count_of_each_color(tlbr_histogram_per_pixel, enable_count_with_minus1, 'tlbr')

    def make_histogram_trbl(self, enable_count_with_minus1: bool):
        """
        A histogram on the diagonal from top-right to bottom-left
        """
        outside_color = 10
        trbl_histograms = []
        skewed_image_up = image_skew(self.image, outside_color, SkewDirection.UP)
        for y in range(skewed_image_up.shape[0]):
            row = skewed_image_up[y, :]
            # convert to 2d image
            row_image = np.expand_dims(row, axis=0)
            histogram = Histogram.create_with_image(row_image)
            histogram.remove_color(outside_color)
            # print(f'y={y} histogram={histogram.pretty()}')
            trbl_histograms.append(histogram)
        # show_prediction_result(self.image, skewed_image_up, None)

        trbl_histogram_per_pixel = []
        for y in range(self.height):
            for x in range(self.width):
                trbl_histogram_per_pixel.append(trbl_histograms[self.width - 1 - x + y])
        self.assign_data_with_unique_colors(trbl_histogram_per_pixel, 'trbl')
        self.assign_data_with_count_of_each_color(trbl_histogram_per_pixel, enable_count_with_minus1, 'trbl')

    def make_gravity_draw(self, gravity_draw_directions: list[GravityDrawDirection]):
        for direction in gravity_draw_directions:
            for color in range(10):
                gd_image = image_gravity_draw(self.image, color, direction)
                self.data[f'gravity_draw_direction{direction}_color{color}'] = gd_image.flatten().tolist()

    def make_erosion(self, pixel_connectivity_list: list[PixelConnectivity]):
        for pixel_connectivity in pixel_connectivity_list:
            erosion_image = image_erosion_multicolor(self.image, pixel_connectivity)
            self.data[f'image_erosion_multicolor_connectivity{pixel_connectivity}'] = erosion_image.flatten().tolist()

    def make_shape3x3_operations(self, shape3x3_operations: list[Shape3x3Operation], lookaround_size: int):
        image_operation_list = []
        for operation in shape3x3_operations:
            if operation == Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_ALL9:
                image2 = ImageShape3x3Histogram.number_of_unique_colors_all9(self.image)
                image_operation_list.append((image2, operation))
            elif operation == Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER:
                image2 = ImageShape3x3Histogram.number_of_unique_colors_around_center(self.image)
                image_operation_list.append((image2, operation))
            elif operation == Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_IN_CORNERS:
                image2 = ImageShape3x3Histogram.number_of_unique_colors_in_corners(self.image)
                image_operation_list.append((image2, operation))
            elif operation == Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4:
                image2 = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(self.image)
                image_operation_list.append((image2, operation))
            elif operation == Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5:
                image2 = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(self.image)
                image_operation_list.append((image2, operation))
            else:
                raise ValueError(f'Unknown shape3x3 operation: {operation}')

        for (image_shape3x3, operation) in image_operation_list:
            assert image_shape3x3.shape == self.image.shape

        for (image_shape3x3, operation) in image_operation_list:
            k = lookaround_size
            n = k * 2 + 1
            for ry in range(n):
                for rx in range(n):
                    values = []
                    for y in range(self.height):
                        for x in range(self.width):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height:
                                values.append(0)
                            else:
                                values.append(image_shape3x3[yy, xx] + 100)
                    self.data[f'image_shape3x3_{operation}_x{rx}_y{ry}'] = values

    def make_mass_compare_adjacent_rowcol(self, steps: list[int]):
        if self.width < 2 or self.height < 2:
            raise ValueError('IMAGE_MASS_COMPARE_ADJACENT_ROWCOL+IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 requires at least 2x2 image. Soft-error.')
        mass_compare_adjacent_rows = image_mass_compare_adjacent_rows(self.image, 0, 1, 2)
        mass_compare_adjacent_rows_height = mass_compare_adjacent_rows.shape[0]
        mass_compare_adjacent_columns = image_mass_compare_adjacent_columns(self.image, 0, 1, 2)
        mass_compare_adjacent_columns_width = mass_compare_adjacent_columns.shape[1]

        for step in steps:
            values_rows_a = []
            values_rows_b = []
            values_columns_a = []
            values_columns_b = []
            for y in range(self.height):
                for x in range(self.width):
                    if y >= step:
                        comp = mass_compare_adjacent_rows[y - step, x]
                    else:
                        comp = 4
                    values_rows_a.append(comp)
                    if x >= step:
                        comp = mass_compare_adjacent_columns[y, x - step]
                    else:
                        comp = 4
                    values_columns_a.append(comp)
                    if y < mass_compare_adjacent_rows_height - step:
                        comp = mass_compare_adjacent_rows[y + step, x]
                    else:
                        comp = 5
                    values_rows_b.append(comp)
                    if x < mass_compare_adjacent_columns_width - step:
                        comp = mass_compare_adjacent_columns[y, x + step]
                    else:
                        comp = 5
                    values_columns_b.append(comp)
            self.data[f'mass_compare_adjacent_rows_a_step{step}'] = values_rows_a
            self.data[f'mass_compare_adjacent_rows_b_step{step}'] = values_rows_b
            self.data[f'mass_compare_adjacent_columns_a_step{step}'] = values_columns_a
            self.data[f'mass_compare_adjacent_columns_b_step{step}'] = values_columns_b

    def make_bounding_boxes_of_each_color(self):
        for color in range(10):
            ignore_colors = []
            for ignore_color in range(10):
                if ignore_color != color:
                    ignore_colors.append(ignore_color)
            rect = find_bounding_box_multiple_ignore_colors(self.image, ignore_colors)
            values = []
            for y in range(self.height):
                for x in range(self.width):
                    is_inside = x >= rect.x and x < rect.x + rect.width and y >= rect.y and y < rect.y + rect.height
                    values.append(int(is_inside))
            self.data[f'bounding_box_of_color{color}'] = values

    def make_bigram_rowcol(self):
        width = self.width
        height = self.height
        bigrams_top_bottom = np.zeros((height-1, width), dtype=np.uint32)
        bigrams_left_right = np.zeros((height, width-1), dtype=np.uint32)
        for y in range(height-1):
            for x in range(width):
                bigrams_top_bottom[y, x] = self.image[y, x] * 10 + self.image[y+1, x]
        for y in range(height):
            for x in range(width-1):
                bigrams_left_right[y, x] = self.image[y, x] * 10 + self.image[y, x+1]

        values_x = []
        values_x_minus1 = []
        for y in range(height):
            for x in range(width):
                if x > 0:
                    values_x_minus1.append(bigrams_left_right[y, x - 1])
                else:
                    values_x_minus1.append(256)
                if x < width - 1:
                    values_x.append(bigrams_left_right[y, x])
                else:
                    values_x.append(257)
        self.data['bigram_left_right_a'] = values_x
        self.data['bigram_left_right_b'] = values_x_minus1

        values_y = []
        values_y_minus1 = []
        for y in range(height):
            for x in range(width):
                if y > 0:
                    values_y_minus1.append(bigrams_top_bottom[y - 1, x])
                else:
                    values_y_minus1.append(256)
                if y < height - 1:
                    values_y.append(bigrams_top_bottom[y, x])
                else:
                    values_y.append(257)
        self.data['bigram_top_bottom_a'] = values_y
        self.data['bigram_top_bottom_b'] = values_y_minus1
