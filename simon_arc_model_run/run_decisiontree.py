import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.image_gravity_move import *
from simon_arc_lab.image_gravity_draw import *
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
task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/692cd3b6.json'
task = Task.load_arcagi1(task_path)
task_id = os.path.splitext(os.path.basename(task_path))[0]
task.metadata_task_id = task_id

#task.show()

def xs_for_input_image(image: int, pair_index: int):
    height, width = image.shape

    ignore_mask = np.zeros_like(image)
    components = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.ALL8, image, ignore_mask)

    # Image with object ids
    object_ids = np.zeros((height, width), dtype=np.uint32)
    object_id_start = (pair_index + 1) * 1000
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

    values_list = []
    for y in range(height):
        for x in range(width):
            values = []
            values.append(pair_index)
            values.append(image[y, x])

            values.append(object_ids[y, x])
            values.append(object_masses[y, x])

            values_list.append(values)
    return values_list

def ys_for_output_image(image: int):
    height, width = image.shape
    values = []
    for y in range(height):
        for x in range(width):
            values.append(image[y, x])
    return values

xs = []
ys = []

for pair_index in range(task.count_examples):
    input_image = task.example_input(pair_index)
    output_image = task.example_output(pair_index)

    input_height, input_width = input_image.shape
    output_height, output_width = output_image.shape
    if input_height != output_height or input_width != output_width:
        raise ValueError('Input and output image must have the same size')
    
    xs_image = xs_for_input_image(input_image, pair_index)
    xs.extend(xs_image)

    ys_image = ys_for_output_image(output_image)
    ys.extend(ys_image)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(xs, ys)

pair_index = 0 
input_image = task.test_input(pair_index)
expected_image = task.test_output(pair_index)
pred_xs = xs_for_input_image(input_image, task.count_examples + pair_index)
expected_ys = ys_for_output_image(task.test_output(pair_index))

result = clf.predict(pred_xs)
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

plt.figure()
tree.plot_tree(clf, filled=True)
plt.show()








