from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_shape3x3_opposite import ImageShape3x3Opposite
from simon_arc_lab.image_shape3x3_center import ImageShape3x3Center
from simon_arc_lab.image_shape3x3_histogram import *
from simon_arc_lab.image_count3x3 import *
from simon_arc_lab.image_erosion_multicolor import image_erosion_multicolor
from simon_arc_lab.image_distort import image_distort
from simon_arc_lab.image_raytrace_probecolor import *
from simon_arc_lab.image_outline import *
from simon_arc_lab.image_gravity_draw import *
from simon_arc_lab.image_skew import *
from simon_arc_lab.image_mass_compare import *
from simon_arc_lab.image_rotate45 import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.find_bounding_box import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_similarity import ImageSimilarity, Feature, FeatureType
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_prediction_result
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
from scipy.stats import entropy
import matplotlib.pyplot as plt
from enum import Enum
import pandas as pd
import numpy as np

class Transformation(Enum):
    DO_NOTHING = 'do_nothing'
    ROTATE_CW = 'rotate_cw'
    ROTATE_CCW = 'rotate_ccw'
    ROTATE_180 = 'rotate_180'
    FLIP_X = 'flip_x'
    FLIP_Y = 'flip_y'
    FLIP_A = 'flip_a'
    FLIP_B = 'flip_b'
    SKEW_UP = 'skew_up'
    SKEW_DOWN = 'skew_down'
    SKEW_LEFT = 'skew_left'
    SKEW_RIGHT = 'skew_right'
    ROTATE_CW_45 = 'rotate_cw_45'
    ROTATE_CCW_45 = 'rotate_ccw_45'

    def apply(self, image: np.array) -> np.array:
        if self == Transformation.DO_NOTHING:
            return image
        elif self == Transformation.ROTATE_CW:
            return image_rotate_cw(image)
        elif self == Transformation.ROTATE_CCW:
            return image_rotate_ccw(image)
        elif self == Transformation.ROTATE_180:
            return image_rotate_180(image)
        elif self == Transformation.FLIP_X:
            return image_flipx(image)
        elif self == Transformation.FLIP_Y:
            return image_flipy(image)
        elif self == Transformation.FLIP_A:
            return image_flip_diagonal_a(image)
        elif self == Transformation.FLIP_B:
            return image_flip_diagonal_b(image)
        elif self == Transformation.SKEW_UP:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.UP)
        elif self == Transformation.SKEW_DOWN:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.DOWN)
        elif self == Transformation.SKEW_LEFT:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.LEFT)
        elif self == Transformation.SKEW_RIGHT:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.RIGHT)
        elif self == Transformation.ROTATE_CW_45:
            fill_color = 10
            return image_rotate_cw_45(image, fill_color)
        elif self == Transformation.ROTATE_CCW_45:
            fill_color = 10
            return image_rotate_ccw_45(image, fill_color)
        else:
            raise ValueError(f'Unknown transformation_index: {self}')


class DecisionTreeFeature(Enum):
    COMPONENT_NEAREST4 = 'component_nearest4'
    COMPONENT_ALL8 = 'component_all8'
    COMPONENT_CORNER4 = 'component_corner4'
    SUPPRESS_CENTER_PIXEL_ONCE = 'suppress_center_pixel_once'
    SUPPRESS_CENTER_PIXEL_LOOKAROUND = 'suppress_center_pixel_lookaround'
    HISTOGRAM_DIAGONAL = 'histogram_diagonal'
    HISTOGRAM_ROWCOL = 'histogram_rowcol'
    HISTOGRAM_VALUE = 'histogram_value'
    IMAGE_MASS_COMPARE_ADJACENT_ROWCOL = 'image_mass_compare_adjacent_rowcol'
    IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 = 'image_mass_compare_adjacent_rowcol2'
    NUMBER_OF_UNIQUE_COLORS_ALL9 = 'number_of_unique_colors_all9'
    NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER = 'number_of_unique_colors_around_center'
    NUMBER_OF_UNIQUE_COLORS_IN_CORNERS = 'number_of_unique_colors_in_corners'
    NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4 = 'number_of_unique_colors_in_diamond4'
    NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5 = 'number_of_unique_colors_in_diamond5'
    ROTATE45 = 'image_rotate45'
    COUNT_NEIGHBORS_WITH_SAME_COLOR = 'count_neighbors_with_same_color'
    EROSION_ALL8 = 'erosion_all8'
    EROSION_NEAREST4 = 'erosion_nearest4'
    EROSION_CORNER4 = 'erosion_corner4'
    EROSION_ROWCOL = 'erosion_rowcol'
    EROSION_DIAGONAL = 'erosion_diagonal'
    ANY_CORNER = 'any_corner'
    CORNER = 'corner'
    ANY_EDGE = 'any_edge'
    CENTER = 'center'
    BOUNDING_BOXES = 'bounding_boxes'
    DISTANCE_INSIDE_OBJECT = 'distance_inside_object'
    GRAVITY_DRAW_TOP_TO_BOTTOM = 'gravity_draw_top_to_bottom'
    GRAVITY_DRAW_BOTTOM_TO_TOP = 'gravity_draw_bottom_to_top'
    GRAVITY_DRAW_LEFT_TO_RIGHT = 'gravity_draw_left_to_right'
    GRAVITY_DRAW_RIGHT_TO_LEFT = 'gravity_draw_right_to_left'
    GRAVITY_DRAW_TOPLEFT_TO_BOTTOMRIGHT = 'gravity_draw_topleft_to_bottomright'
    GRAVITY_DRAW_BOTTOMRIGHT_TO_TOPLEFT = 'gravity_draw_bottomright_to_topleft'
    GRAVITY_DRAW_TOPRIGHT_TO_BOTTOMLEFT = 'gravity_draw_topright_to_bottomleft'
    GRAVITY_DRAW_BOTTOMLEFT_TO_TOPRIGHT = 'gravity_draw_bottomleft_to_topright'
    POSITION_XY0 = 'position_xy0'
    POSITION_XY4 = 'position_xy4'
    OBJECT_ID_RAY_LIST = 'object_id_ray_list'
    IDENTIFY_OBJECT_SHAPE = 'identify_object_shape'
    BIGRAM_ROWCOL = 'bigram_rowcol'
    COLOR_POPULARITY = 'color_popularity'

    @classmethod
    def names_joined_with_comma(cls, features: set['DecisionTreeFeature']) -> str:
        """
        Human readable compact representation of multiple features.
        """
        names_unordered = [feature.name for feature in features]
        names_sorted = sorted(names_unordered)
        return ','.join(names_sorted)

class DecisionTreePredictOutputResult:
    def __init__(self, width: int, height: int, probabilities: np.array):
        self.width = width
        self.height = height
        self.probabilities = probabilities
    
    def confidence_scores(self):
        return np.max(self.probabilities, axis=1)

    def confidence_map(self) -> np.array:
        return self.confidence_scores().reshape(self.height, self.width)

    def low_confidence_pixels(self) -> list[Tuple[int, int]]:
        width = self.width
        confidence_scores = self.confidence_scores()
        confidence_threshold = 0.7  # Adjust this value based on your needs
        low_confidence_indices = np.where(confidence_scores < confidence_threshold)[0]
        xy_positions = [(idx % width, idx // width) for idx in low_confidence_indices]
        # print(f'low_confidence_pixels={xy_positions}')
        return xy_positions
    
    def entropy_scores(self):
        return entropy(self.probabilities.T)
    
    def entropy_map(self) -> np.array:
        return self.entropy_scores().reshape(self.height, self.width)
    
    def high_entropy_pixels(self) -> list[Tuple[int, int]]:
        width = self.width
        entropy_scores = self.entropy_scores()
        entropy_threshold = 0.5  # Adjust based on analysis
        high_entropy_indices = np.where(entropy_scores > entropy_threshold)[0]
        xy_positions = [(idx % width, idx // width) for idx in high_entropy_indices]
        # print(f'high_entropy_pixels={xy_positions}')
        return xy_positions

    def show_confidence_map(self):
        image = self.confidence_map()
        # image = self.entropy_map()
        plt.imshow(image, cmap='hot')
        plt.colorbar()
        plt.title('Prediction Confidence Map')
        plt.show()
    
    def images(self, count: int) -> list[np.array]:
        # Sort probabilities to get class rankings
        sorted_indices_desc = np.argsort(self.probabilities, axis=1)[:, ::-1]

        result_images = []
        for i in range(count):
            # Extract the N'th best prediction
            classes = sorted_indices_desc[:, i]
            # Reshape to image dimensions
            image = classes.reshape(self.height, self.width)
            result_images.append(image)

        return result_images


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

class DecisionTreeUtil:
    @classmethod
    def xs_for_input_image(cls, image: np.array, pair_id: int, features: set[DecisionTreeFeature], is_earlier_prediction: bool) -> dict:
        # print(f'xs_for_input_image: pair_id={pair_id} features={features} is_earlier_prediction={is_earlier_prediction}')
        builder = DataFromImageBuilder(image)

        height, width = image.shape
        pixel_count = width * height 

        outside_color = 10

        builder.make_pair_id(pair_id)
        builder.make_is_earlier_prediction(is_earlier_prediction)

        # Column "center_pixel"
        if DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_ONCE not in features:
            builder.make_center_pixel()

        # Columns "color_popularity"
        if DecisionTreeFeature.COLOR_POPULARITY in features:
            builder.make_color_popularity(1)

        # Position related columns
        if DecisionTreeFeature.POSITION_XY0 in features:
            builder.make_position_xy()

        if DecisionTreeFeature.POSITION_XY4 in features:
            builder.make_position_xy_lookaround(4)

        if DecisionTreeFeature.ANY_EDGE in features:
            builder.make_any_edge()

        if DecisionTreeFeature.ANY_CORNER in features:
            builder.make_any_corner()

        if True:
            builder.make_diagonal_distance()

        if DecisionTreeFeature.CORNER in features:
            builder.make_corner()

        if DecisionTreeFeature.CENTER in features:
            builder.make_center_xy()

        lookaround_size_shape3x3 = 2

        if True:
            suppress_center_pixel_lookaround = DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND in features
            lookaround_size_image_pixel = 1
            builder.make_image_pixel_color(lookaround_size_image_pixel, suppress_center_pixel_lookaround)
    
        component_pixel_connectivity_list = []
        if DecisionTreeFeature.COMPONENT_NEAREST4 in features:
            component_pixel_connectivity_list.append(PixelConnectivity.NEAREST4)
        if DecisionTreeFeature.COMPONENT_ALL8 in features:
            component_pixel_connectivity_list.append(PixelConnectivity.ALL8)
        if DecisionTreeFeature.COMPONENT_CORNER4 in features:
            component_pixel_connectivity_list.append(PixelConnectivity.CORNER4)

        # Connected components
        for component_pixel_connectivity in component_pixel_connectivity_list:
            builder.components(component_pixel_connectivity)

        # Column with shape info
        if DecisionTreeFeature.IDENTIFY_OBJECT_SHAPE in features:
            lookaround_size_shape = 0
            for component_pixel_connectivity in component_pixel_connectivity_list:
                builder.make_object_shape(component_pixel_connectivity, lookaround_size_shape)

        components_list = []
        for component_pixel_connectivity in component_pixel_connectivity_list:
            components = builder.components(component_pixel_connectivity)
            components_list.append((components, component_pixel_connectivity))

        # Assign ids to each object
        object_id_start_list = []
        for component_index in range(len(component_pixel_connectivity_list)):
            object_id_start = (pair_id + 1) * 1000
            if is_earlier_prediction:
                object_id_start += 500
            object_id_start += component_index * 777
            object_id_start_list.append(object_id_start)

        # List with object id images
        object_ids_list = []
        for component_index, component_pixel_connectivity in enumerate(component_pixel_connectivity_list):
            object_id_start = object_id_start_list[component_index]
            object_ids = builder.object_ids(component_pixel_connectivity, object_id_start)
            object_ids_list.append((object_ids, component_pixel_connectivity))

        # Image with object ids
        for component_index, component_pixel_connectivity in enumerate(component_pixel_connectivity_list):
            object_id_start = object_id_start_list[component_index]
            lookaround_size = 0
            builder.make_object_id(component_pixel_connectivity, object_id_start, lookaround_size)

        # Columns related to mass of the object
        for component_index, component_pixel_connectivity in enumerate(component_pixel_connectivity_list):
            object_id_start = object_id_start_list[component_index]
            lookaround_size = 0
            builder.make_object_mass(component_pixel_connectivity, lookaround_size)

        if DecisionTreeFeature.DISTANCE_INSIDE_OBJECT in features:
            for pixel_connectivity in component_pixel_connectivity_list:
                builder.make_distance_inside_object(pixel_connectivity)

        if True:
            lookaround_size = 0
            builder.make_shape3x3_opposite(lookaround_size)

        if True:
            lookaround_size = 0
            builder.make_shape3x3_center(lookaround_size)

        if True:
            builder.make_probe_color_for_all_directions()

        if DecisionTreeFeature.OBJECT_ID_RAY_LIST in features:
            for (object_ids, pixel_connectivity) in object_ids_list:
                builder.make_probe_objectid_for_all_directions(object_ids, pixel_connectivity)

        if True:
            builder.make_outline_all8()

        if True:
            lookaround_size = 1
            builder.make_count_same_color_as_center(lookaround_size)

        if DecisionTreeFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR in features:
            builder.make_count_neighbors_with_same_color()

        enable_count_with_minus1 = DecisionTreeFeature.HISTOGRAM_VALUE in features

        if DecisionTreeFeature.HISTOGRAM_ROWCOL in features:
            builder.make_histogram_row(enable_count_with_minus1)
            builder.make_histogram_column(enable_count_with_minus1)

        if DecisionTreeFeature.HISTOGRAM_DIAGONAL in features:
            builder.make_histogram_tlbr(enable_count_with_minus1)
            builder.make_histogram_trbl(enable_count_with_minus1)

        gravity_draw_directions = []
        if DecisionTreeFeature.GRAVITY_DRAW_TOP_TO_BOTTOM in features:
            gravity_draw_directions.append(GravityDrawDirection.TOP_TO_BOTTOM)
        if DecisionTreeFeature.GRAVITY_DRAW_BOTTOM_TO_TOP in features:
            gravity_draw_directions.append(GravityDrawDirection.TOP_TO_BOTTOM)
        if DecisionTreeFeature.GRAVITY_DRAW_LEFT_TO_RIGHT in features:
            gravity_draw_directions.append(GravityDrawDirection.LEFT_TO_RIGHT)
        if DecisionTreeFeature.GRAVITY_DRAW_RIGHT_TO_LEFT in features:
            gravity_draw_directions.append(GravityDrawDirection.RIGHT_TO_LEFT)
        if DecisionTreeFeature.GRAVITY_DRAW_TOPLEFT_TO_BOTTOMRIGHT in features:
            gravity_draw_directions.append(GravityDrawDirection.TOPLEFT_TO_BOTTOMRIGHT)
        if DecisionTreeFeature.GRAVITY_DRAW_TOPRIGHT_TO_BOTTOMLEFT in features:
            gravity_draw_directions.append(GravityDrawDirection.TOPRIGHT_TO_BOTTOMLEFT)
        if DecisionTreeFeature.GRAVITY_DRAW_BOTTOMLEFT_TO_TOPRIGHT in features:
            gravity_draw_directions.append(GravityDrawDirection.BOTTOMLEFT_TO_TOPRIGHT)
        if DecisionTreeFeature.GRAVITY_DRAW_BOTTOMRIGHT_TO_TOPLEFT in features:
            gravity_draw_directions.append(GravityDrawDirection.BOTTOMRIGHT_TO_TOPLEFT)
        builder.make_gravity_draw(gravity_draw_directions)

        data = builder.data

        erosion_pixel_connectivity_list = []
        if DecisionTreeFeature.EROSION_ALL8 in features:
            erosion_pixel_connectivity_list.append(PixelConnectivity.ALL8)
        if DecisionTreeFeature.EROSION_NEAREST4 in features:
            erosion_pixel_connectivity_list.append(PixelConnectivity.NEAREST4)
        if DecisionTreeFeature.EROSION_CORNER4 in features:
            erosion_pixel_connectivity_list.append(PixelConnectivity.CORNER4)
        if DecisionTreeFeature.EROSION_ROWCOL in features:
            erosion_pixel_connectivity_list.append(PixelConnectivity.LR2)
            erosion_pixel_connectivity_list.append(PixelConnectivity.TB2)
        if DecisionTreeFeature.EROSION_DIAGONAL in features:
            erosion_pixel_connectivity_list.append(PixelConnectivity.TLBR2)
            erosion_pixel_connectivity_list.append(PixelConnectivity.TRBL2)
        for erosion_connectivity in erosion_pixel_connectivity_list:
            erosion_image = image_erosion_multicolor(image, erosion_connectivity)
            data[f'image_erosion_multicolor_connectivity{erosion_connectivity}'] = erosion_image.flatten().tolist()

        shape3x3_images = []
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_ALL9 in features:
            image_number_of_unique_colors_all9 = ImageShape3x3Histogram.number_of_unique_colors_all9(image)
            shape3x3_images.append((image_number_of_unique_colors_all9, 'number_of_unique_colors_all9'))

        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER in features:
            image_number_of_unique_colors_around_center = ImageShape3x3Histogram.number_of_unique_colors_around_center(image)
            shape3x3_images.append((image_number_of_unique_colors_around_center, 'number_of_unique_colors_around_center'))

        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_CORNERS in features:
            image_number_of_unique_colors_in_corners = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
            shape3x3_images.append((image_number_of_unique_colors_in_corners, 'number_of_unique_colors_in_corners'))
        
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4 in features:
            image_number_of_unique_colors_in_diamond4 = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
            shape3x3_images.append((image_number_of_unique_colors_in_diamond4, 'number_of_unique_colors_in_diamond4'))
        
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5 in features:
            image_number_of_unique_colors_in_diamond5 = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(image)
            shape3x3_images.append((image_number_of_unique_colors_in_diamond5, 'number_of_unique_colors_in_diamond5'))

        for (image_shape3x3, transformation_name) in shape3x3_images:
            k = lookaround_size_shape3x3
            n = k * 2 + 1
            for ry in range(n):
                for rx in range(n):
                    values = []
                    for y in range(height):
                        for x in range(width):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                            else:
                                values.append(image_shape3x3[yy, xx] + 100)
                    data[f'image_shape3x3_{transformation_name}_x{rx}_y{ry}'] = values

        mass_compare_adjacent_rows = None
        mass_compare_adjacent_rows_height = 0
        mass_compare_adjacent_columns = None
        mass_compare_adjacent_columns_width = 0
        if (DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL in features) or (DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 in features):
            if width < 2 or height < 2:
                raise ValueError('IMAGE_MASS_COMPARE_ADJACENT_ROWCOL+IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 requires at least 2x2 image. Soft-error.')
            mass_compare_adjacent_rows = image_mass_compare_adjacent_rows(image, 0, 1, 2)
            mass_compare_adjacent_rows_height = mass_compare_adjacent_rows.shape[0]
            mass_compare_adjacent_columns = image_mass_compare_adjacent_columns(image, 0, 1, 2)
            mass_compare_adjacent_columns_width = mass_compare_adjacent_columns.shape[1]

            steps = []
            if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL in features:
                steps.append(1)
            if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 in features:
                steps.append(2)
            
            for step in steps:
                values_rows_a = []
                values_rows_b = []
                values_columns_a = []
                values_columns_b = []
                for y in range(height):
                    for x in range(width):
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
                data[f'image_mass_compare_adjacent_rows_a_step{step}'] = values_rows_a
                data[f'image_mass_compare_adjacent_rows_b_step{step}'] = values_rows_b
                data[f'image_mass_compare_adjacent_columns_a_step{step}'] = values_columns_a
                data[f'image_mass_compare_adjacent_columns_b_step{step}'] = values_columns_b

        if DecisionTreeFeature.BOUNDING_BOXES in features:
            for color in range(10):
                ignore_colors = []
                for ignore_color in range(10):
                    if ignore_color != color:
                        ignore_colors.append(ignore_color)
                rect = find_bounding_box_multiple_ignore_colors(image, ignore_colors)
                values = []
                for y in range(height):
                    for x in range(width):
                        is_inside = x >= rect.x and x < rect.x + rect.width and y >= rect.y and y < rect.y + rect.height
                        values.append(int(is_inside))
                data[f'bounding_box_of_color{color}'] = values

        if DecisionTreeFeature.BIGRAM_ROWCOL in features:
            bigrams_top_bottom = np.zeros((height-1, width), dtype=np.uint32)
            bigrams_left_right = np.zeros((height, width-1), dtype=np.uint32)
            for y in range(height-1):
                for x in range(width):
                    bigrams_top_bottom[y, x] = image[y, x] * 10 + image[y+1, x]
            for y in range(height):
                for x in range(width-1):
                    bigrams_left_right[y, x] = image[y, x] * 10 + image[y, x+1]

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
            data['bigram_left_right_a'] = values_x
            data['bigram_left_right_b'] = values_x_minus1

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
            data['bigram_top_bottom_a'] = values_y
            data['bigram_top_bottom_b'] = values_y_minus1

        count_min, count_max = cls.count_values_xs(data)
        if count_min != count_max:
            raise ValueError(f'The lists must have the same length. However the lists have different lengths. count_min={count_min} count_max={count_max}')

        return data

    @classmethod
    def count_values_xs(cls, xs: dict) -> Tuple[int, int]:
        count_min = 100000000
        count_max = 0
        for key in xs.keys():
            count = len(xs[key])
            count_min = min(count_min, count)
            count_max = max(count_max, count)
        return count_min, count_max

    @classmethod
    def merge_xs_per_pixel(cls, xs0: dict, xs1: dict) -> dict:
        # print("merge_xs_per_pixel before")
        # the keys must be the same
        assert xs0.keys() == xs1.keys()

        # both xs0 and xs1 have the same keys
        assert len(xs0.keys()) == len(xs1.keys())

        xs0_count_min, xs0_count_max = cls.count_values_xs(xs0)
        if xs0_count_min != xs0_count_max:
            raise ValueError(f'Expected same pixel count for lists in xs0 dict, {xs0_count_min} != {xs0_count_max}')

        xs1_count_min, xs1_count_max = cls.count_values_xs(xs1)
        if xs1_count_min != xs1_count_max:
            raise ValueError(f'Expected same pixel count for lists in xs1 dict, {xs1_count_min} != {xs1_count_max}')

        # both xs0 and xs1 should have the same pixel count
        assert xs0_count_min == xs1_count_min

        xs = {}
        # Use different suffixes
        for key in xs0.keys():
            xs[key + "_0"] = xs0[key]
            xs[key + "_1"] = xs1[key]
        # print("merge_xs_per_pixel after")
        return xs
    
    @classmethod
    def xs_for_input_noise_images(cls, refinement_index: int, input_image: np.array, noise_image: np.array, pair_id: int, features: set[DecisionTreeFeature]) -> dict:
        if refinement_index == 0:
            xs = cls.xs_for_input_image(input_image, pair_id, features, False)
        else:
            xs0 = cls.xs_for_input_image(input_image, pair_id, features, False)
            xs1 = cls.xs_for_input_image(noise_image, pair_id, features, True)
            xs = cls.merge_xs_per_pixel(xs0, xs1)
        return xs

    @classmethod
    def ys_for_output_image(cls, image: int) -> list:
        height, width = image.shape
        values = []
        for y in range(height):
            for x in range(width):
                values.append(image[y, x])
        return values

    @classmethod
    def predict_output(cls, task: Task, test_index: int, previous_prediction_image: Optional[np.array], previous_prediction_mask: Optional[np.array], refinement_index: int, noise_level: int, features: set[DecisionTreeFeature]) -> DecisionTreePredictOutputResult:
        if task.has_same_input_output_size_for_all_examples() == False:
            raise ValueError('The decisiontree only works for puzzles where input/output have the same size')
        
        has_previous_prediction_image = previous_prediction_image is not None
        has_previous_prediction_mask = previous_prediction_mask is not None
        if has_previous_prediction_image != has_previous_prediction_mask:
            raise ValueError('Both previous_prediction_image and previous_prediction_mask must be set or be None')

        has_previous = has_previous_prediction_image and has_previous_prediction_mask
        if has_previous:
            if previous_prediction_image.shape != previous_prediction_mask.shape:
                raise ValueError('previous_prediction_image and previous_prediction_mask must have the same size')
            
            test_input_image = task.test_input(test_index)
            if previous_prediction_image.shape != test_input_image.shape:
                raise ValueError('previous_prediction_image and test_input_image must have the same size')

        xs = None
        ys = []

        current_pair_id = 0

        transformation_ids = [
            Transformation.DO_NOTHING,
            # Transformation.ROTATE_CW,
            # Transformation.ROTATE_CCW,
            # Transformation.ROTATE_180,
            # Transformation.FLIP_X,
            # Transformation.FLIP_Y,
            # Transformation.FLIP_A,
            # Transformation.FLIP_B,
            # Transformation.SKEW_UP,
            # Transformation.SKEW_DOWN,
            # Transformation.SKEW_LEFT,
            # Transformation.SKEW_RIGHT,
        ]
        if DecisionTreeFeature.ROTATE45 in features:
            transformation_ids.append(Transformation.ROTATE_CW_45)
            transformation_ids.append(Transformation.ROTATE_CCW_45)

        for pair_index in range(task.count_examples):
            pair_seed = pair_index * 1000 + refinement_index * 10000
            input_image = task.example_input(pair_index)
            output_image = task.example_output(pair_index)

            input_height, input_width = input_image.shape
            output_height, output_width = output_image.shape
            if input_height != output_height or input_width != output_width:
                raise ValueError('Input and output image must have the same size')

            width = input_width
            height = input_height
            positions = []
            for y in range(height):
                for x in range(width):
                    positions.append((x, y))

            random.Random(pair_seed + 1).shuffle(positions)
            # take N percent of the positions
            count_positions = int(len(positions) * noise_level / 100)
            noise_image = output_image.copy()
            for i in range(count_positions):
                x, y = positions[i]
                noise_image[y, x] = input_image[y, x]
            # noise_image = image_distort(noise_image, 1, 25, pair_seed + 1000)

            input_noise_output = []
            transformation_ids_randomized = transformation_ids.copy()
            for transformation in transformation_ids_randomized:
                input_image_mutated = transformation.apply(input_image)
                noise_image_mutated = transformation.apply(noise_image)
                output_image_mutated = transformation.apply(output_image)
                input_noise_output.append((input_image_mutated, noise_image_mutated, output_image_mutated))

            # Shuffle the colors, so it's not always the same color. So all 10 colors gets used.
            h = Histogram.create_with_image(output_image)
            used_colors = h.unique_colors()
            random.Random(pair_seed + 1001).shuffle(used_colors)
            for i in range(10):
                if h.get_count_for_color(i) > 0:
                    continue
                # cycle through the used colors
                first_color = used_colors.pop(0)
                used_colors.append(first_color)

                color_mapping = {
                    first_color: i,
                }
                input_image2 = image_replace_colors(input_image, color_mapping)
                output_image2 = image_replace_colors(output_image, color_mapping)
                noise_image2 = image_replace_colors(noise_image, color_mapping)
                input_noise_output.append((input_image2, noise_image2, output_image2))

            count_mutations = len(input_noise_output)
            for i in range(count_mutations):
                input_image_mutated, noise_image_mutated, output_image_mutated = input_noise_output[i]

                pair_id = current_pair_id * count_mutations + i
                current_pair_id += 1
                xs_image = cls.xs_for_input_noise_images(refinement_index, input_image_mutated, noise_image_mutated, pair_id, features)
                if xs is None:
                    # print(f'iteration 0, setting xs. {type(xs_image)}')
                    xs = xs_image
                else:
                    # print(f'iteration {i}, merging xs. {type(xs_image)}')
                    for key in xs.keys():
                        # print(f'key={key} type={type(xs[key])}')
                        xs[key].extend(xs_image[key])

                ys_image = cls.ys_for_output_image(output_image_mutated)
                ys.extend(ys_image)

        if False:
            # Discard 1/3 of the data
            random.Random(refinement_index).shuffle(xs)
            xs = xs[:len(xs) * 2 // 3]
            random.Random(refinement_index).shuffle(ys)
            ys = ys[:len(ys) * 2 // 3]

        xs_dataframe = pd.DataFrame(xs)

        clf = None
        try:
            clf_inner = DecisionTreeClassifier(random_state=42)
            current_clf = CalibratedClassifierCV(clf_inner, method='isotonic', cv=5)
            current_clf.fit(xs_dataframe, ys)
            clf = current_clf
        except Exception as e:
            print(f'Error: {e}')
        if clf is None:
            print('Falling back to DecisionTreeClassifier')
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(xs_dataframe, ys)

        input_image = task.test_input(test_index)
        height, width = input_image.shape
        noise_image_mutated = input_image.copy()
        if previous_prediction_image is not None:
            noise_image_mutated = previous_prediction_image.copy()

        mask_image = np.ones_like(input_image)
        if previous_prediction_mask is not None:
            mask_image = previous_prediction_mask.copy()
            # colors_available_except_zero = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            # for y in range(height):
            #     for x in range(width):
            #         if mask_image[y, x] > 0:
            #             continue
            #         offset = random.Random(refinement_index + 42 + y * width + x).choice(colors_available_except_zero)
            #         v = noise_image_mutated[y, x]
            #         v = (v + offset) % 10
            #         noise_image_mutated[y, x] = v
            #         # print(f'x={x} y={y} v={v}')
            positions = []
            for y in range(height):
                for x in range(width):
                    if mask_image[y, x] > 0:
                        continue
                    positions.append((x, y))
            random.Random(refinement_index + 42).shuffle(positions)
            positions = positions[:3]
            colors_available = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            color_assign = random.Random(refinement_index + 42).choice(colors_available)
            # for x, y in positions:
            #     noise_image_mutated[y, x] = color_assign

        # Picking a pair_id that has already been used, performs better than picking a new unseen pair_id.
        pair_id = random.Random(refinement_index + 42).randint(0, current_pair_id - 1)
        xs_image = cls.xs_for_input_noise_images(refinement_index, input_image, noise_image_mutated, pair_id, features)
        if False:
            # Randomize the pair_id for the test image, so it doesn't reference a specific example pair
            for i in range(len(xs_image)):
                xs_image[i][0] = random.Random(refinement_index + 42 + i).randint(0, current_pair_id - 1)
        
        # plt.figure()
        # tree.plot_tree(clf, filled=True)
        # plt.show()

        xs_dataframe = pd.DataFrame(xs_image)
        probabilities = clf.predict_proba(xs_dataframe)
        return DecisionTreePredictOutputResult(width, height, probabilities)

    @classmethod
    def validate_output(cls, task: Task, test_index: int, prediction_to_verify: np.array, refinement_index: int, noise_level: int, features: set[DecisionTreeFeature]) -> DecisionTreePredictOutputResult:
        xs = []
        ys = []

        current_pair_id = 0

        transformation_ids = [
            Transformation.DO_NOTHING,
            # Transformation.ROTATE_CW,
            # Transformation.ROTATE_CCW,
            # Transformation.ROTATE_180,
            # Transformation.FLIP_X,
            # Transformation.FLIP_Y,
            # Transformation.FLIP_A,
            # Transformation.FLIP_B,
            # Transformation.SKEW_UP,
            # Transformation.SKEW_DOWN,
            # Transformation.SKEW_LEFT,
            # Transformation.SKEW_RIGHT,
        ]
        if DecisionTreeFeature.ROTATE45 in features:
            transformation_ids.append(Transformation.ROTATE_CW_45)
            transformation_ids.append(Transformation.ROTATE_CCW_45)

        for pair_index in range(task.count_examples):
            pair_seed = pair_index * 1000 + refinement_index * 10000
            input_image = task.example_input(pair_index)
            output_image = task.example_output(pair_index)

            input_height, input_width = input_image.shape
            output_height, output_width = output_image.shape
            if input_height != output_height or input_width != output_width:
                raise ValueError('Input and output image must have the same size')

            width = input_width
            height = input_height
            positions = []
            for y in range(height):
                for x in range(width):
                    positions.append((x, y))

            random.Random(pair_seed + 1).shuffle(positions)
            # take N percent of the positions
            count_positions = int(len(positions) * noise_level / 100)
            noise_image = output_image.copy()
            for i in range(count_positions):
                x, y = positions[i]
                noise_image[y, x] = input_image[y, x]
            noise_image = image_distort(noise_image, 1, 25, pair_seed + 1000)

            input_noise_output = []
            transformation_ids_randomized = transformation_ids.copy()
            for transformation in transformation_ids_randomized:
                input_image_mutated = transformation.apply(input_image)
                noise_image_mutated = transformation.apply(noise_image)
                output_image_mutated = transformation.apply(output_image)
                input_noise_output.append((input_image_mutated, noise_image_mutated, output_image_mutated))

            # Shuffle the colors, so it's not always the same color. So all 10 colors gets used.
            h = Histogram.create_with_image(output_image)
            used_colors = h.unique_colors()
            random.Random(pair_seed + 1001).shuffle(used_colors)
            for i in range(10):
                if h.get_count_for_color(i) > 0:
                    continue
                # cycle through the used colors
                first_color = used_colors.pop(0)
                used_colors.append(first_color)

                color_mapping = {
                    first_color: i,
                }
                input_image2 = image_replace_colors(input_image, color_mapping)
                output_image2 = image_replace_colors(output_image, color_mapping)
                noise_image2 = image_replace_colors(noise_image, color_mapping)
                # input_noise_output.append((input_image2, noise_image2, output_image2))

            count_mutations = len(input_noise_output)
            for i in range(count_mutations):
                input_image_mutated, noise_image_mutated, output_image_mutated = input_noise_output[i]

                pair_id = current_pair_id * count_mutations + i
                current_pair_id += 1
                xs_image = cls.xs_for_input_image(input_image_mutated, pair_id, features, False)

                assert input_image_mutated.shape == output_image_mutated.shape
                height, width = output_image_mutated.shape

                for color in range(10):
                    xs_copy = []
                    ys_copy = []
                    for y in range(height):
                        for x in range(width):
                            value_list_index = y * width + x
                            value_list = list(xs_image[value_list_index]) + [color]
                            xs_copy.append(value_list)
                            is_correct = output_image_mutated[y, x] == color
                            ys_copy.append(int(is_correct))
                    xs.extend(xs_copy)
                    ys.extend(ys_copy)


        if False:
            # Discard 1/3 of the data
            random.Random(refinement_index).shuffle(xs)
            xs = xs[:len(xs) * 2 // 3]
            random.Random(refinement_index).shuffle(ys)
            ys = ys[:len(ys) * 2 // 3]

        # clf = DecisionTreeClassifier(random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(xs, ys)

        input_image = task.test_input(test_index)

        # Picking a pair_id that has already been used, performs better than picking a new unseen pair_id.
        pair_id = random.Random(refinement_index + 42).randint(0, current_pair_id - 1)

        xs_image = cls.xs_for_input_image(input_image, pair_id, features, False)
        assert input_image.shape == prediction_to_verify.shape
        height, width = input_image.shape
        for y in range(height):
            for x in range(width):
                value_list_index = y * width + x
                predicted_color = prediction_to_verify[y, x]
                xs_image[value_list_index].append(predicted_color)

        probabilities = clf.predict_proba(xs_image)
        return DecisionTreePredictOutputResult(width, height, probabilities)

        result = clf.predict(xs_image)

        height, width = input_image.shape
        predicted_image = np.zeros_like(input_image)
        for y in range(height):
            for x in range(width):
                value_raw = result[y * width + x]
                value = int(value_raw)
                if value < 0:
                    value = 0
                if value > 9:
                    value = 9
                predicted_image[y, x] = value

        # plt.figure()
        # tree.plot_tree(clf, filled=True)
        # plt.show()

        return predicted_image
