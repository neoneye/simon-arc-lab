from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_shape3x3_opposite import ImageShape3x3Opposite
from simon_arc_lab.image_shape3x3_center import ImageShape3x3Center
from simon_arc_lab.image_count3x3 import *
from simon_arc_lab.image_distort import image_distort
from simon_arc_lab.image_raytrace_probecolor import *
from simon_arc_lab.image_outline import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.find_bounding_box import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_similarity import ImageSimilarity, Feature, FeatureType
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_prediction_result

class DecisionTreeUtil:

    @classmethod
    def xs_for_input_image(cls, image: int, pair_index: int, is_earlier_prediction: bool):
        height, width = image.shape

        ignore_mask = np.zeros_like(image)
        components = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.ALL8, image, ignore_mask)

        # Image with object ids
        object_ids = np.zeros((height, width), dtype=np.uint32)
        object_id_start = (pair_index + 1) * 1000
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

        values_list = []
        for y in range(height):
            for x in range(width):
                values = []
                values.append(pair_index)
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
