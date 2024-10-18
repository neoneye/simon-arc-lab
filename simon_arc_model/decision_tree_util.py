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
from sklearn import tree
import matplotlib.pyplot as plt
from enum import Enum

class DecisionTreeFeature(Enum):
    HISTOGRAM_DIAGONAL = 'histogram_diagonal'
    HISTOGRAM_ROWCOL = 'histogram_rowcol'
    HISTOGRAM_VALUE = 'histogram_value'
    IMAGE_MASS_COMPARE_ADJACENT_ROWCOL = 'image_mass_compare_adjacent_rowcol'
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
    CORNER = 'corner'
    CENTER = 'center'
    BOUNDING_BOXES = 'bounding_boxes'

class DecisionTreeUtil:
    @classmethod
    def has_same_input_output_size_for_all_examples(cls, task: Task) -> bool:
        for pair_index in range(task.count_examples):
            input_image = task.example_input(pair_index)
            output_image = task.example_output(pair_index)
            if input_image.shape != output_image.shape:
                return False
        return True

    @classmethod
    def xs_for_input_image(cls, image: int, pair_id: int, features: set[DecisionTreeFeature], is_earlier_prediction: bool) -> list:
        height, width = image.shape

        lookaround_size_count_same_color_as_center_with_one_neighbor_nowrap = 1
        lookaround_size = 1
        lookaround_size_object_ids = 0
        lookaround_size_mass = 0

        ignore_mask = np.zeros_like(image)
        components = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.NEAREST4, image, ignore_mask)

        # Image with object ids
        object_ids = np.zeros((height, width), dtype=np.uint32)
        object_id_start = (pair_id + 1) * 1000
        if is_earlier_prediction:
            object_id_start += 500
        for component_index, component in enumerate(components):
            object_id = object_id_start + component_index
            for y in range(height):
                for x in range(width):
                    mask_value = component.mask[y, x]
                    if mask_value == 1:
                        object_ids[y, x] = object_id

        # Image with object mass
        object_masses = np.zeros((height, width), dtype=np.uint32)
        for component_index, component in enumerate(components):
            for y in range(height):
                for x in range(width):
                    mask_value = component.mask[y, x]
                    if mask_value == 1:
                        object_masses[y, x] = component.mass

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

        # gravity_draw_directions = [
        #     GravityDrawDirection.TOP_TO_BOTTOM,
        #     GravityDrawDirection.BOTTOM_TO_TOP,
        #     GravityDrawDirection.LEFT_TO_RIGHT,
        #     GravityDrawDirection.RIGHT_TO_LEFT,
        #     GravityDrawDirection.TOPLEFT_TO_BOTTOMRIGHT,
        #     GravityDrawDirection.BOTTOMRIGHT_TO_TOPLEFT,
        #     GravityDrawDirection.TOPRIGHT_TO_BOTTOMLEFT,
        #     GravityDrawDirection.BOTTOMLEFT_TO_TOPRIGHT,
        # ]
        # gravity_background_color = 0
        # gravity_draw_image_list = []
        # for direction in gravity_draw_directions:
        #     gd_image = image_gravity_draw(image, gravity_background_color, direction)
        #     gravity_draw_image_list.append(gd_image)

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
        if DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL in features:
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

        values_list = []
        for y in range(height):
            for x in range(width):
                values = []
                values.append(pair_id)
                # values.append(x)
                # values.append(y)
                values.append(image[y, x])

                if is_earlier_prediction:
                    values.append(0)
                else:
                    values.append(1)

                x_rev = width - x - 1
                y_rev = height - y - 1

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

                n = lookaround_size
                for dy in range(-n, n * 2):
                    for dx in range(-n, n * 2):
                        if dx == 0 and dy == 0:
                            continue
                        xx = x + dx
                        yy = y + dy
                        if xx < 0 or xx >= width or yy < 0 or yy >= height:
                            values.append(outside_color)
                        else:
                            values.append(image[yy, xx])

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

                k = lookaround_size_mass
                n = k * 2 + 1
                for ry in range(n):
                    for rx in range(n):
                        xx = x + rx - k
                        yy = y + ry - k
                        if xx < 0 or xx >= width or yy < 0 or yy >= height:
                            values.append(0)
                        else:
                            values.append(object_masses[yy, xx])

                values.append(image_shape3x3_opposite[y, x])
                for i in range(3):
                    values.append((image_shape3x3_opposite[y, x] >> i) & 1)

                values.append(image_shape3x3_center[y, x])
                for i in range(8):
                    values.append((image_shape3x3_center[y, x] >> i) & 1)

                for image_ray in image_ray_list:
                    values.append(image_ray[y, x])

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
                
                # for gd_image in gravity_draw_image_list:
                #     values.append(gd_image[y, x])

                for erosion_image in erosion_image_list:
                    values.append(erosion_image[y, x])

                for image_shape3x3 in shape3x3_images:
                    values.append(image_shape3x3[y, x] + 100)

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
    def transform_image(cls, image: np.array, transformation_index: int) -> np.array:
        if transformation_index == 0:
            return image
        elif transformation_index == 1:
            return image_rotate_cw(image)
        elif transformation_index == 2:
            return image_rotate_ccw(image)
        elif transformation_index == 3:
            return image_rotate_180(image)
        elif transformation_index == 4:
            return image_flipx(image)
        elif transformation_index == 5:
            return image_flipy(image)
        elif transformation_index == 6:
            return image_flip_diagonal_a(image)
        elif transformation_index == 7:
            return image_flip_diagonal_b(image)
        elif transformation_index == 8:
            fill_color = 10
            return image_rotate_cw_45(image, fill_color)
        elif transformation_index == 9:
            fill_color = 10
            return image_rotate_ccw_45(image, fill_color)
        else:
            raise ValueError(f'Unknown transformation_index: {transformation_index}')

    @classmethod
    def predict_output(cls, task: Task, test_index: int, previous_prediction: Optional[np.array], refinement_index: int, noise_level: int, features: set[DecisionTreeFeature]) -> np.array:
        xs = []
        ys = []

        current_pair_id = 0

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
            transform_count = 8
            if DecisionTreeFeature.ROTATE45 in features:
                transform_count = 10
            for i in range(transform_count):
                input_image_mutated = cls.transform_image(input_image, i)
                noise_image_mutated = cls.transform_image(noise_image, i)
                output_image_mutated = cls.transform_image(output_image, i)
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

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(xs, ys)

        input_image = task.test_input(test_index)
        noise_image_mutated = input_image.copy()
        if previous_prediction is not None:
            noise_image_mutated = previous_prediction.copy()

        # Picking a pair_id that has already been used, performs better than picking a new unseen pair_id.
        pair_id = random.Random(refinement_index + 42).randint(0, current_pair_id - 1)
        xs_image = cls.xs_for_input_noise_images(refinement_index, input_image, noise_image_mutated, pair_id, features)
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
