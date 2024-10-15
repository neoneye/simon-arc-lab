import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_distort import image_distort
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_model.decision_tree_util import DecisionTreeUtil

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# works with these
task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/b6afb2da.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/6d75e8bb.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/a5313dff.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/bb43febb.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/c0f76784.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/b60334d2.json'
# argh, almost correct
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/d364b489.json'
# struggling with shape issues
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/692cd3b6.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/a699fb00.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/a65b410d.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/4612dd53.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/c97c0139.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/9772c176.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/bdad9b1f.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/d9f24cd1.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/db3e9e38.json'
# struggling with color issues
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/e76a88a6.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/aba27056.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/b782dc8a.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/22168020.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/b548a754.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/e76a88a6.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/b8cdaf2b.json'


task = Task.load_arcagi1(task_path)
task_id = os.path.splitext(os.path.basename(task_path))[0]
task.metadata_task_id = task_id

#task.show()


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
        noise_image = output_image.copy()
        for i in range(count_positions):
            x, y = positions[i]
            noise_image[y, x] = input_image[y, x]
        noise_image = image_distort(noise_image, 1, 25, pair_seed + 1000)

        input_noise_output = []
        for i in range(8):
            input_image_mutated = DecisionTreeUtil.transform_image(input_image, i)
            noise_image_mutated = DecisionTreeUtil.transform_image(noise_image, i)
            output_image_mutated = DecisionTreeUtil.transform_image(output_image, i)
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

            if refinement_index == 0:
                xs_image = DecisionTreeUtil.xs_for_input_image(input_image_mutated, pair_index * count_mutations + i, is_earlier_prediction = False)
                xs.extend(xs_image)
            else:
                xs_image0 = DecisionTreeUtil.xs_for_input_image(input_image_mutated, pair_index * count_mutations + i, is_earlier_prediction = False)
                xs_image1 = DecisionTreeUtil.xs_for_input_image(noise_image_mutated, pair_index * count_mutations + i, is_earlier_prediction = True)
                xs_image = DecisionTreeUtil.merge_xs_per_pixel(xs_image0, xs_image1)
                xs.extend(xs_image)

            ys_image = DecisionTreeUtil.ys_for_output_image(output_image_mutated)
            ys.extend(ys_image)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(xs, ys)

    pair_index = 0
    test_pair_index = task.count_examples + pair_index
    input_image = task.test_input(pair_index)
    noise_image_mutated = input_image.copy()
    if last_predicted_output is not None:
        noise_image_mutated = last_predicted_output.copy()
    expected_image = task.test_output(pair_index)

    if refinement_index == 0:
        xs_image = DecisionTreeUtil.xs_for_input_image(input_image, test_pair_index * 8, is_earlier_prediction = False)
    else:
        xs_image0 = DecisionTreeUtil.xs_for_input_image(input_image, test_pair_index * 8, is_earlier_prediction = False)
        xs_image1 = DecisionTreeUtil.xs_for_input_image(noise_image_mutated, test_pair_index * 8, is_earlier_prediction = True)
        xs_image = DecisionTreeUtil.merge_xs_per_pixel(xs_image0, xs_image1)
    expected_ys = DecisionTreeUtil.ys_for_output_image(task.test_output(pair_index))

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

