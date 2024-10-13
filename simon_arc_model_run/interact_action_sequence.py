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
from simon_arc_lab.image_tile_template import image_tile_template
from simon_arc_lab.find_bounding_box import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_similarity import ImageSimilarity, Feature, FeatureType
from simon_arc_lab.task_similarity import TaskSimilarity

def apply_action_to_image(image: np.array, inventory: dict, s: list) -> Tuple[np.array, dict]:
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
    elif s == 'userecta':
        rect = inventory.get('recta', None)
        if rect is None:
            raise Exception("No recta in inventory")
        inventory['current_rect'] = rect
    elif s == 'userectb':
        rect = inventory.get('rectb', None)
        if rect is None:
            raise Exception("No rectb in inventory")
        inventory['current_rect'] = rect
    elif s == 'userectc':
        rect = inventory.get('rectc', None)
        if rect is None:
            raise Exception("No rectc in inventory")
        inventory['current_rect'] = rect
    elif s == 'userectd':
        rect = inventory.get('rectd', None)
        if rect is None:
            raise Exception("No rectd in inventory")
        inventory['current_rect'] = rect
    elif s == 'drawrectinside':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        current_image = image_rect_inside(current_image, rect, current_color)
    elif s == 'drawrectoutside':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        current_image = image_rect_outside(current_image, rect, current_color)
    elif s == 'crop':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        current_image = current_image[(rect.y):(rect.y + rect.height), (rect.x):(rect.x + rect.width)]
    elif s == 'copyrect':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        image = current_image[(rect.y):(rect.y + rect.height), (rect.x):(rect.x + rect.width)]
        inventory['current_image'] = image
    elif s == 'paste':
        rect = inventory.get('current_rect', None)
        if rect is None:
            raise Exception("No current_rect in inventory")
        image = inventory.get('current_image', None)
        if image is None:
            raise Exception("No current_image in inventory")
        if image.shape != (rect.height, rect.width):
            raise Exception("Image size does not match the rectangle size")
        current_image[(rect.y):(rect.y + rect.height), (rect.x):(rect.x + rect.width)] = image
    elif s == 'setimagea':
        inventory['imagea'] = current_image.copy()
    elif s == 'setimageb':
        inventory['imageb'] = current_image.copy()
    elif s == 'setimagec':
        inventory['imagec'] = current_image.copy()
    elif s == 'setimaged':
        inventory['imaged'] = current_image.copy()
    elif s == 'useimagea':
        image = inventory.get('imagea', None)
        if image is None:
            raise Exception("No imagea in inventory")
        inventory['current_image'] = image.copy()
    elif s == 'useimageb':
        image = inventory.get('imageb', None)
        if image is None:
            raise Exception("No imageb in inventory")
        inventory['current_image'] = image.copy()
    elif s == 'useimagec':
        image = inventory.get('imagec', None)
        if image is None:
            raise Exception("No imagec in inventory")
        inventory['current_image'] = image.copy()
    elif s == 'useimaged':
        image = inventory.get('imaged', None)
        if image is None:
            raise Exception("No imaged in inventory")
        inventory['current_image'] = image.copy()
    elif s == 'loadimage':
        image = inventory.get('current_image', None)
        if image is None:
            raise Exception("No current_image in inventory")
        current_image = image.copy()
    elif s == 'saveimage':
        inventory['current_image'] = current_image.copy()
    elif s == 'collapse':
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        current_image = image_collapse_color(current_image, current_color)
    elif s == 'tile':
        imagea = inventory.get('imagea', None)
        if imagea is None:
            raise Exception("No imagea in inventory")
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        def callback(tile, layout, x, y):
            color_map = {0: layout[y, x], 1: current_color}
            return image_replace_colors(tile, color_map)
        current_image = image_tile_template(imagea, current_image, callback)
    elif s == 'maskinvert':
        current_image = np.where(current_image != 0, 0, 1)
    elif s == 'all8':
        ignore_mask = np.zeros_like(current_image)
        current_objects = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.ALL8, current_image, ignore_mask)
        inventory['current_objects'] = current_objects
    elif s == 'firstobject':
        current_objects = inventory.get('current_objects', None)
        if current_objects is None:
            raise Exception("No current_objects in inventory")
        if len(current_objects) == 0:
            raise Exception("No objects found")
        object = current_objects[0]
        current_image = object.mask.copy()
    elif s == 'deleteobjectswithcolor':
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        current_objects = inventory.get('current_objects', None)
        if current_objects is None:
            raise Exception("No current_objects in inventory")
        current_objects = [object for object in current_objects if object.color != current_color]
        inventory['current_objects'] = current_objects
    elif s == 'gravitydraw_bottom_to_top':
        current_color = inventory.get('current_color', None)
        if current_color is None:
            raise Exception("No current_color in inventory")
        current_image = image_gravity_draw(current_image, current_color, GravityDrawDirection.BOTTOM_TO_TOP)
    else:
        raise Exception(f"Unknown action: {s}")
    return (current_image, inventory)

def apply_actions_to_task(task: Task, action_list: list) -> Task:
    buffer_image_list = []
    for pair_index in range(task.count()):
        input_image = task.input_images[pair_index]
        buffer_image_list.append(input_image.copy())

    inventory_dict_list = []
    for pair_index in range(task.count()):
        inventory_dict_list.append({})
            
    for action in action_list:
        for pair_index in range(task.count()):
            new_output_image, new_inventory_dict = apply_action_to_image(
                buffer_image_list[pair_index], 
                inventory_dict_list[pair_index], 
                action
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

Actions:
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
userecta: set current_rect to rectangle 'a'
userectb: set current_rect to rectangle 'b'
userectc: set current_rect to rectangle 'c'
userectd: set current_rect to rectangle 'd'
drawrectinside: draw current_color inside current rectangle
drawrectoutside: draw current_color outside current rectangle
crop: crop current image to current rectangle
copyrect: copy pixels from current_rectangle to inventory named 'current_image'
paste: paste pixels from inventory named 'image' inside the area specified by 'current_rectangle'
setimagea: set image 'a' to current_image
setimageb: set image 'b' to current_image
setimagec: set image 'c' to current_image
setimaged: set image 'd' to current_image
useimagea: set current_image to image 'a'
useimageb: set current_image to image 'b'
useimagec: set current_image to image 'c'
useimaged: set current_image to image 'd'
loadimage: load current_image from the inventory current_image
saveimage: save current_image in the inventory current_image
collapse: collapse a color
tile: create a repeated pattern using image 'a' as the base tile and image 'b' as the tile layout
maskinvert: convert non-zero pixels to zero and zero pixels to one.
all8: enumerate connected components with PixelConnectivity.ALL8
firstobject: set current_image to the first object found in the inventory current_objects
deleteobjectswithcolor: delete objects with the current_color from the inventory current_objects
gravitydraw_bottom_to_top: apply gravity in the up direction and draw a trail of pixels
"""

# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/009d5c81.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/00dbd492.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/0692e18c.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/9f27f097.json'
# task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation/12997ef3.json'
task_path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training/22168020.json'
original_task = Task.load_arcagi1(task_path)
task_id = os.path.splitext(os.path.basename(task_path))[0]
original_task.metadata_task_id = task_id


action_list_9f27f097 = [
    'setimagea',
    'color0', 'bb2', 'setrecta', 
    'mpc', 'drawrectinside', 
    'mpc', 'bb1', 'copyrect',
    'loadimage',
    'fx',
    'setimageb',
    'useimagea', 
    'loadimage',
    'userecta', 
    'useimageb',
    'paste' 
]

action_list_12997ef3 = [
    'setimaged', 
    'color1', 'bb2', 'crop', 'maskinvert',
    'setimagea', 
    'useimaged', 
    'loadimage', 
    'color0', 'drawrectinside', 'color0', 'bb1', 'crop', 'collapse',
    'color0',
    'tile',
]

action_list_22168020 = [
    'all8', 'color0', 'deleteobjectswithcolor', 
    'firstobject',
    'color1',
    'bb2',
    'color0',
    'gravitydraw_bottom_to_top',
    'color0',
    'drawrectoutside',
]

available_actions = [
    'cw', 'ccw', '180', 'fx', 'fy', 'fa', 'fb', 'mu', 'md', 'ml', 'mr',
    'x2', 'x3', 'x4', 'x5', 'y2', 'y3', 'y4', 'y5', 'xy2', 'xy3', 'xy4', 'xy5',
    'mpc', 'lpc', 'bb1', 'bb2',
    'color0', 'color1', 'color2', 'color3', 'color4', 'color5', 'color6', 'color7', 'color8', 'color9',
    'setrecta', 'setrectb', 'setrectc', 'setrectd',
    'userecta', 'userectb', 'userectc', 'userectd',
    'drawrectinside', 'drawrectoutside',
    'crop',
    'copyrect',
    'paste',
    'setimagea', 'setimageb', 'setimagec', 'setimaged',
    'useimagea', 'useimageb', 'useimagec', 'useimaged',
    'loadimage',
    'saveimage',
    'collapse',
    'tile',
    'maskinvert',
    'all8',
    'firstobject',
    'deleteobjectswithcolor',
    'gravitydraw_up',
]

action_list = []
replay_action_list = []
# replay_action_list = action_list_9f27f097
# replay_action_list = action_list_12997ef3
replay_action_list = action_list_22168020

current_task = apply_actions_to_task(original_task, [])
# if len(replay_action_list) > 0:
#     current_task.show()
for action_index, action in enumerate(replay_action_list):
    print(f"Applying action: {action}")
    action_list.append(action)
    current_task = apply_actions_to_task(original_task, action_list)
    current_task.metadata_task_id = f'{task_id} {action}, {action_index+1} of {len(replay_action_list)}'
    # current_task.show()

current_task.show()

for i in range(100):
    print(f"action_list: {action_list}")
    value = input("Please enter command:\n")
    if len(value) == 0:
        print(available_commands)
        continue

    if value in available_actions:
        action_list.append(value)
        current_task.metadata_task_id = f'{task_id} {action}'
        current_task = apply_actions_to_task(original_task, action_list)
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
        action_list.pop()
        continue

    if value == 'q':
        break

    print(f"Unknown command: {value}\n\nAvailable commands:\n{available_commands}")
