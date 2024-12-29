from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.pixel_connectivity import PixelConnectivity
from simon_arc_lab.image_gravity_draw import GravityDrawDirection
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_similarity import ImageSimilarity, Feature, FeatureType
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_prediction_result
from .data_from_image_builder import DataFromImageBuilder, Shape3x3Operation
from .image_augmentation_operation import ImageAugmentationOperation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
from scipy.stats import entropy
import matplotlib.pyplot as plt
from enum import Enum
import pandas as pd
import numpy as np

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


class DecisionTreeUtil:
    @classmethod
    def xs_for_input_image(cls, image: np.array, pair_id: int, features: set[DecisionTreeFeature], is_earlier_prediction: bool) -> dict:
        # print(f'xs_for_input_image: pair_id={pair_id} features={features} is_earlier_prediction={is_earlier_prediction}')
        builder = DataFromImageBuilder(image)

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
        builder.make_erosion(erosion_pixel_connectivity_list)

        shape3x3_operations = []
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_ALL9 in features:
            shape3x3_operations.append(Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_ALL9)
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER in features:
            shape3x3_operations.append(Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_AROUND_CENTER)
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_CORNERS in features:
            shape3x3_operations.append(Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_IN_CORNERS)
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4 in features:
            shape3x3_operations.append(Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4)
        if DecisionTreeFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5 in features:
            shape3x3_operations.append(Shape3x3Operation.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5)
        lookaround_size_shape3x3 = 2
        builder.make_shape3x3_operations(shape3x3_operations, lookaround_size_shape3x3)

        if (DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL in features) or (DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 in features):
            steps = []
            if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL in features:
                steps.append(1)
            if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2 in features:
                steps.append(2)
            builder.make_mass_compare_adjacent_rowcol(steps)

        if DecisionTreeFeature.BOUNDING_BOXES in features:
            builder.make_bounding_boxes_of_each_color()

        if DecisionTreeFeature.BIGRAM_ROWCOL in features:
            builder.make_bigram_rowcol()

        data = builder.data

        count_min, count_max = cls.count_values_xs(data)
        if count_min != count_max:
            raise ValueError(f'The lists must have the same length. However the lists have different lengths. count_min={count_min} count_max={count_max}')
        # print(f'number of keys: {len(data.keys())}  number of values: {count_min}')

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
            ImageAugmentationOperation.DO_NOTHING,
            # ImageAugmentationOperation.ROTATE_CW,
            # ImageAugmentationOperation.ROTATE_CCW,
            # ImageAugmentationOperation.ROTATE_180,
            # ImageAugmentationOperation.FLIP_X,
            # ImageAugmentationOperation.FLIP_Y,
            # ImageAugmentationOperation.FLIP_A,
            # ImageAugmentationOperation.FLIP_B,
            # ImageAugmentationOperation.SKEW_UP,
            # ImageAugmentationOperation.SKEW_DOWN,
            # ImageAugmentationOperation.SKEW_LEFT,
            # ImageAugmentationOperation.SKEW_RIGHT,
        ]
        if DecisionTreeFeature.ROTATE45 in features:
            transformation_ids.append(ImageAugmentationOperation.ROTATE_CW_45)
            transformation_ids.append(ImageAugmentationOperation.ROTATE_CCW_45)

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
            ImageAugmentationOperation.DO_NOTHING,
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
            transformation_ids.append(ImageAugmentationOperation.ROTATE_CW_45)
            transformation_ids.append(ImageAugmentationOperation.ROTATE_CCW_45)

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
