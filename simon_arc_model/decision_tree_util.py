from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_shape3x3_opposite import ImageShape3x3Opposite
from simon_arc_lab.image_shape3x3_center import ImageShape3x3Center
from simon_arc_lab.image_count3x3 import *
from simon_arc_lab.image_erosion_multicolor import image_erosion_multicolor
from simon_arc_lab.image_distort import image_distort
from simon_arc_lab.image_raytrace_probecolor import *
from simon_arc_lab.image_outline import *
from simon_arc_lab.image_gravity_draw import *
from simon_arc_lab.image_skew import *
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

class DecisionTreeUtil:

    @classmethod
    def xs_for_input_image(cls, image: int, pair_id: int, features: set[DecisionTreeFeature], is_earlier_prediction: bool) -> list:
        height, width = image.shape

        ignore_mask = np.zeros_like(image)
        components = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.ALL8, image, ignore_mask)

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
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                image_same = count_same_color_as_center_with_one_neighbor_nowrap(image, dx, dy)
                image_same_list.append(image_same)

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

        # erosion_pixel_connectivity_list = [
        #     PixelConnectivity.ALL8,
        #     PixelConnectivity.NEAREST4,
        #     PixelConnectivity.CORNER4,
        #     PixelConnectivity.LR2,
        #     PixelConnectivity.TB2,
        #     PixelConnectivity.TLBR2,
        #     PixelConnectivity.TRBL2,
        # ]
        # erosion_image_list = []
        # for erosion_connectivity in erosion_pixel_connectivity_list:
        #     erosion_image = image_erosion_multicolor(image, erosion_connectivity)
        #     erosion_image_list.append(erosion_image)

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

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        xx = x + dx
                        yy = y + dy
                        if xx < 0 or xx >= width or yy < 0 or yy >= height:
                            values.append(outside_color)
                        else:
                            values.append(image[yy, xx])

                values.append(object_ids[y, x])
                values.append(object_masses[y, x])

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

                # for erosion_image in erosion_image_list:
                #     values.append(erosion_image[y, x])


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
            for i in range(8):
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
