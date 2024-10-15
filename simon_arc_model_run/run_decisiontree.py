import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_shape3x3_opposite import ImageShape3x3Opposite
from simon_arc_lab.image_shape3x3_center import ImageShape3x3Center
from simon_arc_lab.image_distort import image_distort
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

# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/22168020.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/692cd3b6.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/9772c176.json'
task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/c97c0139.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/4612dd53.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/6d75e8bb.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/a5313dff.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/a699fb00.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/a65b410d.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/aba27056.json'
task = Task.load_arcagi1(task_path)
task_id = os.path.splitext(os.path.basename(task_path))[0]
task.metadata_task_id = task_id

#task.show()

def xs_for_input_image(image: int, pair_index: int, is_earlier_prediction: bool):
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

            for i in range(3):
                values.append((image_shape3x3_opposite[y, x] >> i) & 1)

            for i in range(8):
                values.append((image_shape3x3_center[y, x] >> i) & 1)

            values_list.append(values)
    return values_list

def merge_xs_per_pixel(xs_list0: list, xs_list1: list) -> list:
    xs_list = []
    assert len(xs_list0) == len(xs_list1)
    for i in range(len(xs_list0)):
        xs = xs_list0[i] + xs_list1[i]
        xs_list.append(xs)
    return xs_list

def ys_for_output_image(image: int):
    height, width = image.shape
    values = []
    for y in range(height):
        for x in range(width):
            values.append(image[y, x])
    return values

def transform_image(image: np.array, transformation_index: int) -> np.array:
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

noise_levels = [95, 90, 85, 80, 75, 70, 65]
number_of_refinements = len(noise_levels)
last_predicted_output = None
for refinement_index in range(number_of_refinements):
    noise_level = noise_levels[refinement_index]
    print(f"Refinement {refinement_index+1}/{number_of_refinements} noise_level={noise_level}")

    xs = []
    ys = []

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
        noisy_image = output_image.copy()
        for i in range(count_positions):
            x, y = positions[i]
            noisy_image[y, x] = input_image[y, x]
        noisy_image = image_distort(noisy_image, 1, 25, pair_seed + 1000)

        for i in range(8):
            input_image_mutated = transform_image(input_image, i)
            output_image_mutated = transform_image(output_image, i)
            noisy_image_mutated = transform_image(noisy_image, i)

            if refinement_index == 0:
                xs_image = xs_for_input_image(input_image_mutated, pair_index * 8 + i, is_earlier_prediction = False)
                xs.extend(xs_image)
            else:
                xs_image0 = xs_for_input_image(input_image_mutated, pair_index * 8 + i, is_earlier_prediction = False)
                xs_image1 = xs_for_input_image(noisy_image_mutated, pair_index * 8 + i, is_earlier_prediction = True)
                xs_image = merge_xs_per_pixel(xs_image0, xs_image1)
                xs.extend(xs_image)

            ys_image = ys_for_output_image(output_image_mutated)
            ys.extend(ys_image)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(xs, ys)

    pair_index = 0
    test_pair_index = task.count_examples + pair_index
    input_image = task.test_input(pair_index)
    noisy_image_mutated = input_image.copy()
    if last_predicted_output is not None:
        noisy_image_mutated = last_predicted_output.copy()
    expected_image = task.test_output(pair_index)

    if refinement_index == 0:
        xs_image = xs_for_input_image(input_image, test_pair_index * 8, is_earlier_prediction = False)
    else:
        xs_image0 = xs_for_input_image(input_image, test_pair_index * 8, is_earlier_prediction = False)
        xs_image1 = xs_for_input_image(noisy_image_mutated, test_pair_index * 8, is_earlier_prediction = True)
        xs_image = merge_xs_per_pixel(xs_image0, xs_image1)
    expected_ys = ys_for_output_image(task.test_output(pair_index))

    result = clf.predict(xs_image)
    #print(result)

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

    show_prediction_result(input_image, predicted_image, expected_image, title = task.metadata_task_id)

    last_predicted_output = predicted_image.copy()
    # plt.figure()
    # tree.plot_tree(clf, filled=True)
    # plt.show()

