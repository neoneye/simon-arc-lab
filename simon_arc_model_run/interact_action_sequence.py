import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.image_util import *
from simon_arc_lab.image_similarity import ImageSimilarity

def apply_manipulations_to_image(image: np.array, manipulation_list: list) -> np.array:
    current_image = image.copy()
    for s in manipulation_list:
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
        else:
            raise Exception(f"Unknown manipulation: {s}")
    return current_image

def apply_manipulations_to_task(task: Task, manipulation_list: list) -> Task:
    current_task = task.clone()
    for pair_index in range(task.count()):
        input_image = task.input_images[pair_index]
        predicted_output_image = apply_manipulations_to_image(input_image, manipulation_list)
        current_task.output_images[pair_index] = predicted_output_image
        if pair_index >= task.count_examples:
            continue
        output_image = task.output_images[pair_index]
        image_similarity = ImageSimilarity.create_with_images(output_image, predicted_output_image)
        score = image_similarity.jaccard_index()
        print(f"pair: {pair_index} score: {score}")
    return current_task


available_commands = """
q: quit
s: show current task
so: show original task
u: undo

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
"""

task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/009d5c81.json'
task = Task.load_arcagi1(task_path)
# task.show()

manipulation_list = ['cw']

available_manipulations = ['cw', 'ccw', '180', 'fx', 'fy', 'fa', 'fb', 'mu', 'md', 'ml', 'mr']

current_task = apply_manipulations_to_task(task, manipulation_list)
for i in range(100):
    print(f"manipulation_list: {manipulation_list}")
    value = input("Please enter command:\n")
    if len(value) == 0:
        print(available_commands)
        continue

    if value in available_manipulations:
        manipulation_list.append(value)
        current_task = apply_manipulations_to_task(task, manipulation_list)
        continue

    if value == 'so':
        task.show()
        continue

    if value == 's':
        current_task.show()
        continue

    if value == 'u':
        manipulation_list.pop()
        continue

    if value == 'q':
        break

    print(f"Unknown command: {value}\n\nAvailable commands:\n{available_commands}")
