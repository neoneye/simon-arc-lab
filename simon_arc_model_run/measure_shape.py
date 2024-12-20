from datetime import datetime
import sys
import os
import json
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.shape import *
from simon_arc_lab.pixel_connectivity import PixelConnectivity
from simon_arc_lab.connected_component import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_string_representation import image_to_string
from simon_arc_lab.show_prediction_result import show_multiple_images

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    ('ARC-AGI', 'arcagi', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data')),
    # ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('ARC-AGI', 'arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('arc-dataset-tama', 'tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('Mini-ARC', 'miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('ConceptARC', 'conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('ARC-AGI', 'testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

task_ids_of_interest = [
    # '0b17323b',
    '08573cc6',
    # '6c434453',
    # '21f83797',
    # '13713586',
    # '1c02dbbe',
    # '29700607',
    # '1a2e2828',
]

class PythonImageBuilder:
    """
    Generate Python code with rectangles

    image=np.zeros((10,20),dtype=np.uint8)
    image[y:y+height, x:x+width] = color # fill a rectangle
    image[y, x:x+width] = color # fill multiple columns
    image[:, x:x+width] = color # fill an entire column
    image[y, x] = color # set a single pixel
    """
    def __init__(self, original_image: np.array, background_color: Optional[int], name: Optional[str]):
        height, width = original_image.shape
        self.original_image = original_image
        self.original_image_height = height
        self.original_image_width = width

        if background_color is None:
            histogram = Histogram.create_with_image(original_image)
            background_color = histogram.most_popular_color()
        self.background_color = background_color

        if name is None:
            name = "image"
        self.name = name

        self.lines = []
        if background_color is None or background_color == 0:
            self.lines.append(f"{name}=np.zeros(({height},{width}),dtype=np.uint8)")
        else:
            self.lines.append(f"{name}=np.full(({height},{width}),{background_color},dtype=np.uint8)")

    def rectangle(self, x: int, y: int, width: int, height: int, color: int):
        assert width >= 1 and height >= 1
        assert x >= 0 and y >= 0

        if color == self.background_color:
            return
        
        if width == 1:
            x_str = str(x)
        else:
            x_str = f"{x}:{x+width}"
        if width == self.original_image_width and x == 0:
            x_str = ":"

        if height == 1:
            y_str = str(y)
        else:
            y_str = f"{y}:{y+height}"
        if height == self.original_image_height and y == 0:
            y_str = ":"

        code = f"{self.name}[{y_str},{x_str}]={color}"
        self.lines.append(code)

    def get_code(self) -> str:
        return "\n".join(self.lines)


def analyze_image(image: np.array, image_id: str) -> list[str]:
    background_color = 0
    connectivity = PixelConnectivity.NEAREST4
    ignore_color = background_color
    ignore_mask = (image == ignore_color).astype(np.uint8)
    connected_components = ConnectedComponent.find_objects_with_ignore_mask_inner(connectivity, image, ignore_mask)

    print(f"Number of connected components: {len(connected_components)}")

    python_image_builder = PythonImageBuilder(image, background_color, image_id)

    for connected_component_item in connected_components:
        # print(f"Connected component item: {connected_component_item}")
        mask = connected_component_item.mask
        color = connected_component_item.color
        shape = image_find_shape(mask, verbose=False)
        if shape is None:
            print(f"Connected component item: {connected_component_item}")
            print(image_to_string(mask))
            # print(mask.tolist())
            continue
        print(f"Shape: {shape}")

        if isinstance(shape, SolidRectangleShape):
            python_image_builder.rectangle(
                shape.rectangle.x, 
                shape.rectangle.y, 
                shape.rectangle.width, 
                shape.rectangle.height, 
                color
            )
        else:
            print(f"Unhandled shape: {shape}")
    
    return python_image_builder.lines

def analyze_task(task: Task, test_index: int):
    rows = []
    rows.append("")
    rows.append("Solve this ARC puzzle.")
    rows.append("")
    rows.append("```python")
    for i in range(task.count_examples):
        rows.append(f"# pair {i}")
        input_image = task.example_input(i)
        input_image_id = f"input{i}"
        input_rows = analyze_image(input_image, input_image_id)
        rows.extend(input_rows)

        output_image = task.example_output(i)
        output_image_id = f"output{i}"
        output_rows = analyze_image(output_image, output_image_id)
        rows.extend(output_rows)
    
    if True:
        rows.append(f"# pair {task.count_examples}")
        input_image = task.test_input(test_index)
        input_image_id = f"input{task.count_examples}"
        input_rows = analyze_image(input_image, input_image_id)
        rows.extend(input_rows)
    
    output_image_id = f"output{task.count_examples}"
    rows.append(f"{output_image_id}=PREDICT THIS!")

    rows.append("```")
    rows.append("")
    rows.append(f"Just populate the `{output_image_id}` with the correct values.")
    rows.append("")
    s = '\n'.join(rows)
    print(s)

def verify_task(task: Task, test_index: int):

    # output3 = np.zeros((13,13), dtype=np.uint8)
    # output3[2,0:11] = 2
    # output3[2:13,11] = 8
    # output3[4,1:9] = 2
    # output3[4:12,9] = 8
    # output3[5:13,1] = 8
    # output3[5,3:5] = 2
    # output3[5,6] = 1
    # output3[6:11,3] = 8
    # output3[7,4:10] = 2
    # output3[9,5:11] = 8
    # output3[12,2:12] = 2

    # output3=np.zeros((13,13),dtype=np.uint8)

    # # Define horizontal lines for Label A (2)
    # horizontal_rows = [1, 3, 5, 8, 10]
    # for row in horizontal_rows:
    #     output3[row, :] = 2

    # # Define vertical lines for Label B (8)
    # vertical_cols = [2, 4, 6, 8, 10]
    # for col in vertical_cols:
    #     output3[:, col] = 8

    # # Set the intersection point to Label C (1)
    # output3[5,6] = 1

    output3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 2, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 8],
        [0, 8, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 8],
        [0, 8, 0, 2, 2, 0, 0, 0, 8, 0, 0, 0, 8],
        [0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8],
        [0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8],
        [0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8],
        [0, 8, 0, 8, 2, 2, 2, 2, 2, 0, 0, 0, 8],
        [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 8]], dtype=np.uint8)

    predicted_output_image = output3
    expected_output_image = task.test_output(test_index)
    input_image = task.test_input(test_index)

    is_correct = np.array_equal(predicted_output_image, expected_output_image)

    status = "correct" if is_correct else "incorrect"

    title_image_list = [
        ('arc', 'input', input_image),
        ('arc', 'target', expected_output_image),
        ('arc', 'predicted', predicted_output_image),
    ]
    title = f"{task.metadata_task_id} status: {status}"
    show_multiple_images(title_image_list, title=title)


number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.keep_tasks_with_id(set(task_ids_of_interest), verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    pbar = tqdm(taskset.tasks, desc=f"Processing tasks in {groupname}", dynamic_ncols=True)
    for task in pbar:
        task_id = task.metadata_task_id
        pbar.set_postfix_str(f"Task: {task_id}")

        for test_index in range(task.count_tests):
            analyze_task(task, test_index)
            # verify_task(task, test_index)
            # filename = f'{task_id}_test{test_index}_prompt.md'
            # filepath = os.path.join(save_dir, filename)
            # with open(filepath, 'w') as f:
            #     f.write(prompt)
