from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_gaussian_splatting import ImageGaussianSplatting
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
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from scipy.stats import entropy
import matplotlib.pyplot as plt
from enum import Enum
import math

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
    GAUSSIAN_SPLATTING = 'gaussian_splatting'

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


class DecisionTreeUtil:
    @classmethod
    def xs_for_input_image(cls, image: np.array, pair_id: int, features: set[DecisionTreeFeature], is_earlier_prediction: bool) -> list:
        # print(f'xs_for_input_image: pair_id={pair_id} features={features} is_earlier_prediction={is_earlier_prediction}')
        height, width = image.shape

        lookaround_size_count_same_color_as_center_with_one_neighbor_nowrap = 1
        lookaround_size_image_pixel = 1
        lookaround_size_shape = 0
        lookaround_size_object_ids = 0
        lookaround_size_mass = 0
        lookaround_size_shape3x3 = 0
        lookaround_size_shape3x3_center = 0
        lookaround_size_shape3x3_opposite = 0

        component_pixel_connectivity_list = []
        if DecisionTreeFeature.COMPONENT_NEAREST4 in features:
            component_pixel_connectivity_list.append(PixelConnectivity.NEAREST4)
        if DecisionTreeFeature.COMPONENT_ALL8 in features:
            component_pixel_connectivity_list.append(PixelConnectivity.ALL8)
        if DecisionTreeFeature.COMPONENT_CORNER4 in features:
            component_pixel_connectivity_list.append(PixelConnectivity.CORNER4)

        # Connected components
        ignore_mask = np.zeros_like(image)
        components_list = []
        for component_pixel_connectivity in component_pixel_connectivity_list:
            # get a binary mask, where the image is 5 and the rest is not 5
            image2 = np.where(image != 0, 1, 0)
            components = ConnectedComponent.find_objects_with_ignore_mask_inner(component_pixel_connectivity, image2, ignore_mask)
            components_list.append(components)

        object_shape_list = []
        if DecisionTreeFeature.IDENTIFY_OBJECT_SHAPE in features:
            for component_index, components in enumerate(components_list):
                object_shape = np.zeros((height, width), dtype=np.uint32)
                for component_index, component in enumerate(components):
                    rect = find_bounding_box_ignoring_color(component.mask, 0)
                    rect_mass = rect.width * rect.height
                    value = 0
                    if component.mass == rect_mass:
                        if rect.width > rect.height:
                            value = 1
                        elif rect.width < rect.height:
                            value = 2
                        else:
                            value = 3
                    for y in range(height):
                        for x in range(width):
                            mask_value = component.mask[y, x]
                            if mask_value == 1:
                                object_shape[y, x] = value
                object_shape_list.append(object_shape)

        if DecisionTreeFeature.GAUSSIAN_SPLATTING in features:
            image_gaussian_splatting_angle = np.zeros((height, width), dtype=np.float32)
            image_gaussian_splatting_spread_primary = np.zeros((height, width), dtype=np.float32)
            image_gaussian_splatting_spread_secondary = np.zeros((height, width), dtype=np.float32)
            image_gaussian_splatting_center_x = np.zeros((height, width), dtype=np.float32)
            image_gaussian_splatting_center_y = np.zeros((height, width), dtype=np.float32)
            for component_index, components in enumerate(components_list):
                for component in components:
                    if not np.any(component.mask):
                        print('Skipping component with no mask')
                        continue
                    igs = ImageGaussianSplatting(component.mask)
                    if np.isnan(igs.angle) or np.isinf(igs.angle):
                        print(f'Skipping component with invalid angle. angle={igs.angle}')
                        continue
                    if np.isnan(igs.spread_primary) or np.isinf(igs.spread_primary):
                        print(f'Skipping component with invalid spread_primary. spread_primary={igs.spread_primary}')
                        continue
                    if np.isnan(igs.spread_secondary) or np.isinf(igs.spread_secondary):
                        print(f'Skipping component with invalid spread_secondary. spread_secondary={igs.spread_secondary}')
                        continue
                    # loop over non-zero pixels
                    for x in range(width):
                        for y in range(height):
                            mask_value = component.mask[y, x]
                            if mask_value == 1:
                                image_gaussian_splatting_angle[y, x] = igs.angle
                                image_gaussian_splatting_spread_primary[y, x] = igs.spread_primary
                                image_gaussian_splatting_spread_secondary[y, x] = igs.spread_secondary
                                image_gaussian_splatting_center_x[y, x] = igs.x_c
                                image_gaussian_splatting_center_y[y, x] = igs.y_c

        # Image with object ids
        object_ids_list = []
        for component_index, components in enumerate(components_list):
            object_ids = np.zeros((height, width), dtype=np.uint32)
            object_id_start = (pair_id + 1) * 1000
            if is_earlier_prediction:
                object_id_start += 500
            object_id_start += component_index * 777
            for component_index, component in enumerate(components):
                object_id = object_id_start + component_index
                for y in range(height):
                    for x in range(width):
                        mask_value = component.mask[y, x]
                        if mask_value == 1:
                            object_ids[y, x] = object_id
            object_ids_list.append(object_ids)

        # Image with object mass
        object_masses_list = []
        for components in components_list:
            object_masses = np.zeros((height, width), dtype=np.uint32)
            for component_index, component in enumerate(components):
                for y in range(height):
                    for x in range(width):
                        mask_value = component.mask[y, x]
                        if mask_value == 1:
                            object_masses[y, x] = component.mass
            object_masses_list.append(object_masses)

        object_distance_list = []
        if DecisionTreeFeature.DISTANCE_INSIDE_OBJECT in features:
            for component_index, components in enumerate(components_list):
                object_distance_topleft = np.zeros((height, width), dtype=np.uint32)
                object_distance_topright = np.zeros((height, width), dtype=np.uint32)
                object_distance_bottomleft = np.zeros((height, width), dtype=np.uint32)
                object_distance_bottomright = np.zeros((height, width), dtype=np.uint32)
                object_distance_top = np.zeros((height, width), dtype=np.uint32)
                object_distance_bottom = np.zeros((height, width), dtype=np.uint32)
                object_distance_left = np.zeros((height, width), dtype=np.uint32)
                object_distance_right = np.zeros((height, width), dtype=np.uint32)
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
                object_distance_list.append(object_distance_topleft)
                object_distance_list.append(object_distance_topright)
                object_distance_list.append(object_distance_bottomleft)
                object_distance_list.append(object_distance_bottomright)
                object_distance_list.append(object_distance_top)
                object_distance_list.append(object_distance_bottom)
                object_distance_list.append(object_distance_left)
                object_distance_list.append(object_distance_right)

        image_shape3x3_opposite = ImageShape3x3Opposite.apply(image)
        image_shape3x3_center = ImageShape3x3Center.apply(image)

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
        image_ray_list = []
        for direction in ray_directions:
            image_ray = image_raytrace_probecolor_direction(image, outside_color, direction)
            image_ray_list.append(image_ray)

        object_id_ray_list = []
        if DecisionTreeFeature.OBJECT_ID_RAY_LIST in features:
            for object_ids in object_ids_list:
                for direction in ray_directions:
                    object_id_outside_color = 0
                    object_id_ray = image_raytrace_probecolor_direction(object_ids, object_id_outside_color, direction)
                    object_id_ray_list.append(object_id_ray)

        the_image_outline_all8 = image_outline_all8(image)

        image_same_list = []
        n = lookaround_size_count_same_color_as_center_with_one_neighbor_nowrap
        for dy in range(-n, n * 2):
            for dx in range(-n, n * 2):
                if dx == 0 and dy == 0:
                    continue
                image_same = count_same_color_as_center_with_one_neighbor_nowrap(image, dx, dy)
                image_same_list.append(image_same)

        if DecisionTreeFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR in features:
            image_count_neightbors_with_same_color = count_neighbors_with_same_color_nowrap(image)
        else:
            image_count_neightbors_with_same_color = None

        full_histogram = Histogram.create_with_image(image)
        most_popular_color_set = set(full_histogram.most_popular_color_list())
        least_popular_color_set = set(full_histogram.least_popular_color_list())

        row_histograms = []
        column_histogram = []
        if DecisionTreeFeature.HISTOGRAM_ROWCOL in features:
            for y in range(height):
                row = image[y, :]
                # convert to 2d image
                row_image = np.expand_dims(row, axis=0)
                histogram = Histogram.create_with_image(row_image)
                row_histograms.append(histogram)

            for x in range(width):
                column = image[:, x]
                # convert to 2d image
                column_image = np.expand_dims(column, axis=0)
                histogram = Histogram.create_with_image(column_image)
                column_histogram.append(histogram)

        tlbr_histograms = []
        trbl_histograms = []
        if DecisionTreeFeature.HISTOGRAM_DIAGONAL in features:
            skewed_image_down = image_skew(image, outside_color, SkewDirection.DOWN)
            for y in range(skewed_image_down.shape[0]):
                row = skewed_image_down[y, :]
                # convert to 2d image
                row_image = np.expand_dims(row, axis=0)
                histogram = Histogram.create_with_image(row_image)
                histogram.remove_color(outside_color)
                # print(f'y={y} histogram={histogram.pretty()}')
                tlbr_histograms.append(histogram)
            # show_prediction_result(image, skewed_image_down, None)

            skewed_image_up = image_skew(image, outside_color, SkewDirection.UP)
            for y in range(skewed_image_up.shape[0]):
                row = skewed_image_up[y, :]
                # convert to 2d image
                row_image = np.expand_dims(row, axis=0)
                histogram = Histogram.create_with_image(row_image)
                histogram.remove_color(outside_color)
                # print(f'y={y} histogram={histogram.pretty()}')
                trbl_histograms.append(histogram)
            # show_prediction_result(image, skewed_image_up, None)

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
        gravity_draw_image_list = []
        for direction in gravity_draw_directions:
            for color in range(10):
                gd_image = image_gravity_draw(image, color, direction)
                gravity_draw_image_list.append(gd_image)

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
        erosion_image_list = []
        for erosion_connectivity in erosion_pixel_connectivity_list:
            erosion_image = image_erosion_multicolor(image, erosion_connectivity)
            erosion_image_list.append(erosion_image)

        shape3x3_images = []
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_ALL9 in features:
            image_number_of_unique_colors_all9 = ImageShape3x3Histogram.number_of_unique_colors_all9(image)
            shape3x3_images.append(image_number_of_unique_colors_all9)

        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER in features:
            image_number_of_unique_colors_around_center = ImageShape3x3Histogram.number_of_unique_colors_around_center(image)
            shape3x3_images.append(image_number_of_unique_colors_around_center)

        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_CORNERS in features:
            image_number_of_unique_colors_in_corners = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
            shape3x3_images.append(image_number_of_unique_colors_in_corners)
        
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4 in features:
            image_number_of_unique_colors_in_diamond4 = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
            shape3x3_images.append(image_number_of_unique_colors_in_diamond4)
        
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5 in features:
            image_number_of_unique_colors_in_diamond5 = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(image)
            shape3x3_images.append(image_number_of_unique_colors_in_diamond5)

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

        bounding_box_list = []
        if DecisionTreeFeature.BOUNDING_BOXES in features:
            for color in range(10):
                ignore_colors = []
                for ignore_color in range(10):
                    if ignore_color != color:
                        ignore_colors.append(ignore_color)
                rect = find_bounding_box_multiple_ignore_colors(image, ignore_colors)
                bounding_box_list.append(rect)

        bigrams_top_bottom = None
        bigrams_left_right = None
        if DecisionTreeFeature.BIGRAM_ROWCOL in features:
            bigrams_top_bottom = np.zeros((height-1, width), dtype=np.uint32)
            bigrams_left_right = np.zeros((height, width-1), dtype=np.uint32)
            for y in range(height-1):
                for x in range(width):
                    bigrams_top_bottom[y, x] = image[y, x] * 10 + image[y+1, x]
            for y in range(height):
                for x in range(width-1):
                    bigrams_left_right[y, x] = image[y, x] * 10 + image[y, x+1]

        values_list = []
        for y in range(height):
            for x in range(width):
                values = []
                values.append(pair_id)

                if DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_ONCE not in features:
                    values.append(image[y, x])

                if DecisionTreeFeature.COLOR_POPULARITY in features:
                    k = 1
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                                values.append(0)
                                values.append(0)
                            else:
                                color = image[yy, xx]
                                is_most_popular = color in most_popular_color_set
                                values.append(int(is_most_popular))
                                is_least_popular = color in least_popular_color_set
                                values.append(int(is_least_popular))
                                is_medium_popular = is_most_popular == False and is_least_popular == False
                                values.append(int(is_medium_popular))

                if is_earlier_prediction:
                    values.append(0)
                else:
                    values.append(1)

                x_rev = width - x - 1
                y_rev = height - y - 1

                if DecisionTreeFeature.POSITION_XY0 in features:
                    values.append(x)
                    values.append(y)
                    values.append(x_rev)
                    values.append(y_rev)
                if DecisionTreeFeature.POSITION_XY4 in features:
                    for i in range(4):
                        j = i + 1
                        values.append(x + j)
                        values.append(x - j)
                        values.append(y + j)
                        values.append(y - j)
                        values.append(x_rev + j)
                        values.append(x_rev - j)
                        values.append(y_rev + j)
                        values.append(y_rev - j)

                if DecisionTreeFeature.ANY_EDGE in features:
                    is_edge = x == 0 or x_rev == 0 or y == 0 or y_rev == 0
                    values.append(int(is_edge))

                if DecisionTreeFeature.ANY_CORNER in features:
                    is_corner = (x == 0 and y == 0) or (x == 0 and y_rev == 0) or (x_rev == 0 and y == 0) or (x_rev == 0 and y_rev == 0)
                    values.append(int(is_corner))

                steps = [1, 3, 7]
                for step in steps:
                    if (x + y) & step > 0:
                        values.append(1)
                    else:
                        values.append(0)

                    if (x_rev + y) & step > 0:
                        values.append(1)
                    else:
                        values.append(0)

                    if (x + y_rev) & step > 0:
                        values.append(1)
                    else:
                        values.append(0)

                    if (x_rev + y_rev) & step > 0:
                        values.append(1)
                    else:
                        values.append(0)

                if DecisionTreeFeature.BOUNDING_BOXES in features:
                    for rect in bounding_box_list:
                        is_inside = x >= rect.x and x < rect.x + rect.width and y >= rect.y and y < rect.y + rect.height
                        values.append(int(is_inside))

                if DecisionTreeFeature.CORNER in features:
                    corner_topleft = x == 0 and y == 0
                    corner_topright = x_rev == 0 and y == 0
                    corner_bottomleft = x == 0 and y_rev == 0
                    corner_bottomright = x_rev == 0 and y_rev == 0
                    corner_values = [corner_topleft, corner_topright, corner_bottomleft, corner_bottomright]
                    # convert bools to 0 or 1
                    corner_values = [int(x) for x in corner_values]
                    values.extend(corner_values)
                
                if DecisionTreeFeature.CENTER in features:
                    is_center_column = abs(x - x_rev) < 2
                    is_center_row = abs(y - y_rev) < 2
                    values.append(int(is_center_column))
                    values.append(int(is_center_row))

                suppress_center_pixel_lookaround = DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND in features
                k = lookaround_size_image_pixel
                n = k * 2 + 1
                for ry in range(n):
                    for rx in range(n):
                        if suppress_center_pixel_lookaround and rx == k and ry == k:
                            continue
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= width or yy < 0 or yy >= height:
                            values.append(outside_color)
                        else:
                            values.append(image[yy, xx])

                for object_shape in object_shape_list:
                    k = lookaround_size_shape
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                            else:
                                values.append(object_shape[yy, xx])

                for object_ids in object_ids_list:
                    k = lookaround_size_object_ids
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                            else:
                                values.append(object_ids[yy, xx])

                for object_masses in object_masses_list:
                    k = lookaround_size_mass
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                mass = 0
                            else:
                                mass = object_masses[yy, xx]
                            values.append(mass)
                            # values.append(int(np.sqrt(mass) * 10.0))

                for object_distance in object_distance_list:
                    values.append(object_distance[y, x])

                if True:
                    k = lookaround_size_shape3x3_opposite
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                            else:
                                values.append(image_shape3x3_opposite[yy, xx])

                for i in range(3):
                    values.append((image_shape3x3_opposite[y, x] >> i) & 1)

                if True:
                    k = lookaround_size_shape3x3_center
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                            else:
                                values.append(image_shape3x3_center[yy, xx])

                for i in range(8):
                    values.append((image_shape3x3_center[y, x] >> i) & 1)

                for image_ray in image_ray_list:
                    values.append(image_ray[y, x])

                for object_ids in object_id_ray_list:
                    values.append(object_ids[y, x])

                is_outline = the_image_outline_all8[y, x]
                if is_outline == 1:
                    values.append(100)
                else:
                    values.append(-100)

                for image_same in image_same_list:
                    is_same = image_same[y, x] == 1
                    if is_same:
                        values.append(100)
                    else:
                        values.append(-100)
                
                for gd_image in gravity_draw_image_list:
                    values.append(gd_image[y, x])

                for erosion_image in erosion_image_list:
                    values.append(erosion_image[y, x])

                for image_shape3x3 in shape3x3_images:
                    k = lookaround_size_shape3x3
                    n = k * 2 + 1
                    for ry in range(n):
                        for rx in range(n):
                            xx = x + rx - k
                            yy = y + ry - k
                            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                                values.append(0)
                            else:
                                values.append(image_shape3x3[yy, xx] + 100)

                if DecisionTreeFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR in features:
                    values.append(image_count_neightbors_with_same_color[y, x])

                histograms = []
                if DecisionTreeFeature.HISTOGRAM_ROWCOL in features:
                    histograms.append(row_histograms[y])
                    histograms.append(column_histogram[x])

                if DecisionTreeFeature.HISTOGRAM_DIAGONAL in features:
                    tlbr_histogram = tlbr_histograms[x + y]
                    trbl_histogram = trbl_histograms[width - 1 - x + y]
                    histograms.append(tlbr_histogram)
                    histograms.append(trbl_histogram)

                for histogram in histograms:
                    unique_colors = histogram.unique_colors()
                    number_of_unique_colors = len(unique_colors)
                    values.append(number_of_unique_colors)
                    for i in range(10):
                        count = histogram.get_count_for_color(i)
                        values.append(count)

                        if DecisionTreeFeature.HISTOGRAM_VALUE in features:
                            if count > 0:
                                values.append(i)
                            else:
                                values.append(-1)

                        if i in unique_colors:
                            values.append(1)
                        else:
                            values.append(0)

                if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL in features:
                    if y > 0:
                        comp = mass_compare_adjacent_rows[y - 1, x]
                    else:
                        comp = 4
                    values.append(comp)
                    if x > 0:
                        comp = mass_compare_adjacent_columns[y, x - 1]
                    else:
                        comp = 4
                    values.append(comp)
                    if y < mass_compare_adjacent_rows_height - 1:
                        comp = mass_compare_adjacent_rows[y + 1, x]
                    else:
                        comp = 5
                    values.append(comp)
                    if x < mass_compare_adjacent_columns_width - 1:
                        comp = mass_compare_adjacent_columns[y, x + 1]
                    else:
                        comp = 5
                    values.append(comp)

                if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 in features:
                    if y > 1:
                        comp = mass_compare_adjacent_rows[y - 2, x]
                    else:
                        comp = 4
                    values.append(comp)
                    if x > 1:
                        comp = mass_compare_adjacent_columns[y, x - 2]
                    else:
                        comp = 4
                    values.append(comp)
                    if y < mass_compare_adjacent_rows_height - 2:
                        comp = mass_compare_adjacent_rows[y + 2, x]
                    else:
                        comp = 5
                    values.append(comp)
                    if x < mass_compare_adjacent_columns_width - 2:
                        comp = mass_compare_adjacent_columns[y, x + 2]
                    else:
                        comp = 5
                    values.append(comp)

                if bigrams_top_bottom is not None:
                    if y > 0:
                        values.append(bigrams_top_bottom[y - 1, x])
                    else:
                        values.append(256)
                    if y < height - 1:
                        values.append(bigrams_top_bottom[y, x])
                    else:
                        values.append(257)
                if bigrams_left_right is not None:
                    if x > 0:
                        values.append(bigrams_left_right[y, x - 1])
                    else:
                        values.append(256)
                    if x < width - 1:
                        values.append(bigrams_left_right[y, x])
                    else:
                        values.append(257)

                if DecisionTreeFeature.GAUSSIAN_SPLATTING in features:
                    value_angle = image_gaussian_splatting_angle[y, x]
                    # values.append(math.cos(value_angle))
                    # values.append(math.sin(value_angle))
                    # values.append(image_gaussian_splatting_spread_primary[y, x])
                    # values.append(image_gaussian_splatting_spread_secondary[y, x])
                    # values.append(image_gaussian_splatting_center_x[y, x])
                    # values.append(image_gaussian_splatting_center_y[y, x])
                    a = image_gaussian_splatting_spread_primary[y, x]
                    b = image_gaussian_splatting_spread_secondary[y, x]
                    a05 = a / 2
                    a2 = a * 2
                    b05 = b / 2
                    b2 = b * 2
                    values.append(a < b05)
                    values.append(a >= b05 and a < b)
                    values.append(a >= b and a < b2)
                    values.append(a >= b2)
                    values.append(b < a05)
                    values.append(b >= a05 and b < a)
                    values.append(b >= a and b < a2)
                    values.append(b >= a2)

                values_list.append(values)
        return values_list

    @classmethod
    def merge_xs_per_pixel(cls, xs_list0: list, xs_list1: list) -> list:
        xs_list = []
        assert len(xs_list0) == len(xs_list1)
        for i in range(len(xs_list0)):
            xs = xs_list0[i] + xs_list1[i]
            xs_list.append(xs)
        return xs_list
    
    @classmethod
    def xs_for_input_noise_images(cls, refinement_index: int, input_image: np.array, noise_image: np.array, pair_id: int, features: set[DecisionTreeFeature]) -> list:
        if refinement_index == 0:
            xs_image = cls.xs_for_input_image(input_image, pair_id, features, False)
        else:
            xs_image0 = cls.xs_for_input_image(input_image, pair_id, features, False)
            xs_image1 = cls.xs_for_input_image(noise_image, pair_id, features, True)
            xs_image = cls.merge_xs_per_pixel(xs_image0, xs_image1)
        return xs_image

    @classmethod
    def ys_for_output_image(cls, image: int):
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

        xs = []
        ys = []

        current_pair_id = 0

        transformation_ids = [
            Transformation.DO_NOTHING,
            Transformation.ROTATE_CW,
            Transformation.ROTATE_CCW,
            Transformation.ROTATE_180,
            Transformation.FLIP_X,
            Transformation.FLIP_Y,
            Transformation.FLIP_A,
            Transformation.FLIP_B,
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
                xs.extend(xs_image)
                ys_image = cls.ys_for_output_image(output_image_mutated)
                ys.extend(ys_image)

        if False:
            # Discard 1/3 of the data
            random.Random(refinement_index).shuffle(xs)
            xs = xs[:len(xs) * 2 // 3]
            random.Random(refinement_index).shuffle(ys)
            ys = ys[:len(ys) * 2 // 3]

        clf = None
        if False:
            try:
                # raise ValueError('SIMON Ignore the AdaBoostClassifier for now')
                # Define multiple "weak" or at least base estimators
                # estimators = [
                #     ('knn', KNeighborsClassifier(n_neighbors=5)),
                #     ('svc', SVC(probability=True, kernel='linear'))
                # ]

                # # The final estimator (meta-learner) that takes predictions of the above as features
                # final_estimator = LogisticRegression()

                # # Create the stacking ensemble
                # current_clf = StackingClassifier(
                #     estimators=estimators,
                #     final_estimator=final_estimator,
                #     cv=5
                # )

                # current_clf = DecisionTreeClassifier(max_depth=1)

                weak_learner = DecisionTreeClassifier(
                    # criterion='gini', 
                    # criterion='entropy', 
                    # criterion='log_loss', 
                    # max_depth=5,
                    max_depth=10,
                    # random_state=42
                )

                # Create an AdaBoost classifier that sequentially adds weak learners
                current_clf = AdaBoostClassifier(
                    estimator=weak_learner,
                    n_estimators=40,
                    learning_rate=0.01,
                    random_state=42,
                    algorithm='SAMME'
                )
                # print(f'Fitting AdaBoostClassifier: {current_clf}')
                # clf_inner = DecisionTreeClassifier(random_state=42)
                # current_clf = CalibratedClassifierCV(clf_inner, method='isotonic', cv=5)
                current_clf.fit(xs, ys)
                # print(f'After fitting AdaBoostClassifier: {current_clf}')
                clf = current_clf
            except Exception as e:
                print(f'Error: {e}')
        if clf is None:
            # print('Falling back to DecisionTreeClassifier')
            clf = DecisionTreeClassifier(random_state=42, max_depth=8)
            clf.fit(xs, ys)

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

        probabilities = clf.predict_proba(xs_image)
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
