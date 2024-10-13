import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.find_bounding_box import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_similarity import ImageSimilarity, Feature, FeatureType
from simon_arc_lab.task_similarity import TaskSimilarity

def apply_manipulation_to_image(image: np.array, inventory: dict, s: list) -> Tuple[np.array, dict]:
    current_image = image.copy()
    if s == 'cw':
        current_image = image_rotate_cw(current_image)
    elif s == 'ccw':
        current_image = image_rotate_ccw(current_image)
    elif s == '180':
        current_image = image_rotate_180(current_image)
    elif s == 'fx':
        current_image = image_flipx(current_image)
    elif s == 'fy':
        current_image = image_flipy(current_image)
    elif s == 'fa':
        current_image = image_flip_diagonal_a(current_image)
    elif s == 'fb':
        current_image = image_flip_diagonal_b(current_image)
    elif s == 'mu':
        current_image = image_translate_wrap(current_image, 0, -1)
    elif s == 'md':
        current_image = image_translate_wrap(current_image, 0, 1)
    elif s == 'ml':
        current_image = image_translate_wrap(current_image, -1, 0)
    elif s == 'mr':
        current_image = image_translate_wrap(current_image, 1, 0)
    elif s == 'x2':
        _, current_image = image_scale(current_image, 'up', 2, 'up', 1)
    elif s == 'x3':
        _, current_image = image_scale(current_image, 'up', 3, 'up', 1)
    elif s == 'x4':
        _, current_image = image_scale(current_image, 'up', 4, 'up', 1)
    elif s == 'x5':
        _, current_image = image_scale(current_image, 'up', 5, 'up', 1)
    elif s == 'y2':
        _, current_image = image_scale(current_image, 'up', 1, 'up', 2)
    elif s == 'y3':
        _, current_image = image_scale(current_image, 'up', 1, 'up', 3)
    elif s == 'y4':
        _, current_image = image_scale(current_image, 'up', 1, 'up', 4)
    elif s == 'y5':
        _, current_image = image_scale(current_image, 'up', 1, 'up', 5)
    elif s == 'xy2':
        _, current_image = image_scale(current_image, 'up', 2, 'up', 2)
    elif s == 'xy3':
        _, current_image = image_scale(current_image, 'up', 3, 'up', 3)
    elif s == 'xy4':
        _, current_image = image_scale(current_image, 'up', 4, 'up', 4)
    elif s == 'xy5':
        _, current_image = image_scale(current_image, 'up', 5, 'up', 5)
    elif s == 'mpc':
        h = Histogram.create_with_image(current_image)
        color = h.most_popular_color()
        if color is not None:
            color = int(color)
        inventory['current_color'] = color
    elif s == 'lpc':
        h = Histogram.create_with_image(current_image)
        color = h.least_popular_color()
        if color is not None:
            color = int(color)
        inventory['current_color'] = int(color)
    elif s == 'bb1':
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        ignore_color = current_color
        rect = find_bounding_box_ignoring_color(current_image, ignore_color)
        inventory['current_rect'] = rect
    elif s == 'bb2':
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        ignore_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ignore_colors.remove(current_color)
        rect = find_bounding_box_multiple_ignore_colors(current_image, ignore_colors)
        inventory['current_rect'] = rect
    elif s == 'color0':
        inventory['current_color'] = 0
    elif s == 'color1':
        inventory['current_color'] = 1
    elif s == 'color2':
        inventory['current_color'] = 2
    elif s == 'color3':
        inventory['current_color'] = 3
    elif s == 'color4':
        inventory['current_color'] = 4
    elif s == 'color5':
        inventory['current_color'] = 5
    elif s == 'color6':
        inventory['current_color'] = 6
    elif s == 'color7':
        inventory['current_color'] = 7
    elif s == 'color8':
        inventory['current_color'] = 8
    elif s == 'color9':
        inventory['current_color'] = 9
    elif s == 'setrecta':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        inventory['recta'] = rect
    elif s == 'setrectb':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        inventory['rectb'] = rect
    elif s == 'setrectc':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        inventory['rectc'] = rect
    elif s == 'setrectd':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        inventory['rectd'] = rect
    elif s == 'drawrect':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        current_image = image_rect(current_image, inventory['recta'], current_color)
    else:
        raise Exception(f"Unknown manipulation: {s}")
    return (current_image, inventory)

def apply_manipulations_to_task(task: Task, manipulation_list: list) -> Task:
    buffer_image_list = []
    for pair_index in range(task.count()):
        input_image = task.input_images[pair_index]
        buffer_image_list.append(input_image.copy())

    inventory_dict_list = []
    for pair_index in range(task.count()):
        inventory_dict_list.append({})
            
    for manipulation in manipulation_list:
        for pair_index in range(task.count()):
            new_output_image, new_inventory_dict = apply_manipulation_to_image(
                buffer_image_list[pair_index], 
                inventory_dict_list[pair_index], 
                manipulation
            )
            buffer_image_list[pair_index] = new_output_image
            inventory_dict_list[pair_index] = new_inventory_dict

    current_task = task.clone()
    for pair_index in range(task.count()):
        new_output_image = buffer_image_list[pair_index]
        current_task.output_images[pair_index] = new_output_image

    # IDEA: the inventory colors, measure how close they are to the missing colors of the output images.
    for pair_index in range(task.count_examples):
        output_image = task.output_images[pair_index]
        new_output_image = current_task.output_images[pair_index]
        image_similarity = ImageSimilarity.create_with_images(output_image, new_output_image)
        score = image_similarity.jaccard_index()
        unsatisfied_features = image_similarity.get_unsatisfied_features()
        issues = []
        if Feature(FeatureType.SAME_WIDTH) in unsatisfied_features:
            issues.append('width')
        if Feature(FeatureType.SAME_HEIGHT) in unsatisfied_features:
            issues.append('height')
        issue_str = ','.join(issues)
        inventory = inventory_dict_list[pair_index]
        print(f"pair: {pair_index} score: {score} issues: {issue_str} inventory: {inventory}")
    return current_task

def print_features(task: Task):
    task_similarity = TaskSimilarity.create_with_task(task)
    print(f"task_similarity summary: {task_similarity.summary()}")
    print(f"task_similarity pair features: {task_similarity.example_pair_feature_set_intersection}")

available_commands = """
Commands:
q: quit
s: show current task
so: show original task
u: undo
pf: print features of current task
pfo: print features of original task

Manipulations:
cw: rotate clockwise
ccw: rotate counter clockwise
180: rotate 180
fx: flip x
fy: flip y
fa: flip diagonal a
fb: flip diagonal b
mu: move up
md: move down
ml: move left
mr: move right
x2: scale x-axis by 2
x3: scale x-axis by 3
x4: scale x-axis by 4
x5: scale x-axis by 5
y2: scale y-axis by 2
y3: scale y-axis by 3
y4: scale y-axis by 4
y5: scale y-axis by 5
xy2: scale x-axis and y-axis by 2
xy3: scale x-axis and y-axis by 3
xy4: scale x-axis and y-axis by 4
xy5: scale x-axis and y-axis by 5
mpc: take most popular color from input image and save in inventory
lpc: take least popular color from input image and save in inventory
bb1: find bounding box ignoring the current color
bb2: find bounding box of the current color
color0: set current_color to 0
color1: set current_color to 1
color2: set current_color to 2
color3: set current_color to 3
color4: set current_color to 4
color5: set current_color to 5
color6: set current_color to 6
color7: set current_color to 7
color8: set current_color to 8
color9: set current_color to 9
setrecta: set rectangle 'a' to current_rect
setrectb: set rectangle 'b' to current_rect
setrectc: set rectangle 'c' to current_rect
setrectd: set rectangle 'd' to current_rect
"""

# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/009d5c81.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/00dbd492.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/0692e18c.json'
task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/9f27f097.json'
original_task = Task.load_arcagi1(task_path)

manipulation_list = []

available_manipulations = [
    'cw', 'ccw', '180', 'fx', 'fy', 'fa', 'fb', 'mu', 'md', 'ml', 'mr',
    'x2', 'x3', 'x4', 'x5', 'y2', 'y3', 'y4', 'y5', 'xy2', 'xy3', 'xy4', 'xy5',
    'mpc', 'lpc', 'bb1', 'bb2',
    'color0', 'color1', 'color2', 'color3', 'color4', 'color5', 'color6', 'color7', 'color8', 'color9',
    'setrecta', 'setrectb', 'setrectc', 'setrectd',
    'drawrect'
]

current_task = apply_manipulations_to_task(original_task, manipulation_list)
for i in range(100):
    print(f"manipulation_list: {manipulation_list}")
    value = input("Please enter command:\n")
    if len(value) == 0:
        print(available_commands)
        continue

    if value in available_manipulations:
        manipulation_list.append(value)
        current_task = apply_manipulations_to_task(original_task, manipulation_list)
        continue

    if value == 'so':
        original_task.show()
        continue

    if value == 's':
        current_task.show()
        continue

    if value == 'pf':
        print_features(current_task)
        continue

    if value == 'pfo':
        print_features(original_task)
        continue

    if value == 'u':
        manipulation_list.pop()
        continue

    if value == 'q':
        break

    print(f"Unknown command: {value}\n\nAvailable commands:\n{available_commands}")
