# Translate the image by 1 pixel, up/down/left/right.
# Determine what kind of translation happened.
#
# It would be easier if I create an entire ARC task json file.
# On the other hand, I need the benchmark ids, so I can determine what the model struggles with.
#
# Present the same input images, but with different transformations.
# so there are examples of up, down, left, right, and the model should determine what happened.
import random
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *

def generate_task(seed: int, dx: int, dy: int, percent_correct: float) -> Task:
    count_example = random.Random(seed + 1).randint(2, 5)
    count_test = random.Random(seed + 2).randint(2, 3)
    task = Task()
    min_width = 1
    max_width = 10
    min_height = 1
    max_height = 10

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced(seed + 1000 + i, min_width, max_width, min_height, max_height)

        transformed_image = image_translate_wrap(input_image, dx, dy)

        height, width = transformed_image.shape
        noise_image = image_create_random_advanced(seed + 1001 + i, width, width, height, height)
        mask = image_create_random_with_two_colors(width, height, 0, 1, percent_correct, seed + 1050 + i)

        output_image = image_mix(mask, transformed_image, noise_image)

        task.append_pair(input_image, output_image, is_example)

    return task

ratios = [0.0, 0.33, 0.5]
for i in range(3):
    ratio = ratios[i]
    task = generate_task(0, 0, 1, ratio)
    task.show()
