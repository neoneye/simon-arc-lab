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
from simon_arc_lab.image_string_representation import image_to_string
from simon_arc_lab.show_prediction_result import show_multiple_images
from simon_arc_lab.python_image_builder import PythonImageBuilder

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
    # '08573cc6',
    # '8403a5d5',
    # '8d510a79',
    '6fa7a44f',
    # '6c434453',
    # '21f83797',
    # '13713586',
    # '1c02dbbe',
    # '29700607',
    # '1a2e2828',
]

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
        test_input_image = task.example_input(i)
        test_input_image_id = f"input{i}"
        test_input_rows = analyze_image(test_input_image, test_input_image_id)
        rows.extend(test_input_rows)

        output_image = task.example_output(i)
        test_output_image_id = f"output{i}"
        output_rows = analyze_image(output_image, test_output_image_id)
        rows.extend(output_rows)
    
    rows.append(f"# pair {task.count_examples}")
    test_input_image = task.test_input(test_index)
    # test_input_image_id = f"input{task.count_examples}"
    test_input_image_id = "test_input"
    test_input_rows = analyze_image(test_input_image, test_input_image_id)
    rows.extend(test_input_rows)
    
    # test_output_image_id = f"output{task.count_examples}"
    test_output_image_id = "test_output"
    rows.append(f"{test_output_image_id}=PREDICT THIS!")

    rows.append("```")
    rows.append("")
    rows.append("Understanding the puzzle before presenting the answer.")
    rows.append("")
    rows.append("Carefully do step-by-step reasoning. Don't hide any reasoning steps.")
    rows.append("")
    rows.append("Analyze the given input-output pairs to identify the transformation pattern.")
    rows.append("")
    rows.append(f"Given the `{test_input_image_id}` array as defined, determine the exact numeric values of each cell in `{test_output_image_id}`.")
    rows.append(f"Do not refer to `{test_input_image_id}` in your answer—just show the final array of digits that result")
    rows.append(f"from applying the same transformation pattern observed in the previous pairs.")
    rows.append("")
    rows.append(f"Return the content of `{test_output_image_id}` as json wrapped in three back quotes, like this:")
    rows.append(f"```json")
    rows.append(f"[[1,2,3],[4,5,6]]")
    rows.append(f"```")
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

    # xoutput3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    #     [0, 2, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 8],
    #     [0, 8, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 8],
    #     [0, 8, 0, 2, 2, 0, 0, 0, 8, 0, 0, 0, 8],
    #     [0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8],
    #     [0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8],
    #     [0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8],
    #     [0, 8, 0, 8, 2, 2, 2, 2, 2, 0, 0, 0, 8],
    #     [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    #     [0, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 8]], dtype=np.uint8)
    
    # output3=np.zeros((13,13),dtype=np.uint8)
    # output3[2,0:11]=2
    # output3[2:13,11]=8
    # output3[4,1:9]=2
    # output3[4:11,9]=8
    # output3[5:13,1]=8
    # output3[6,3:7]=2
    # output3[6,6]=1
    # output3[7:10,3]=8
    # output3[9,4:10]=2
    # output3[11,2:11]=2

    # output3=np.zeros((10,10),dtype=np.uint8)
    # output3[:,2]=1
    # output3[0,3]=5
    # output3[:,4]=1
    # output3[:,6]=1
    # output3[0,7]=5
    # output3[:,8]=1
    # output3[9,5]=5

    # output2=np.zeros((10,10),dtype=np.uint8)
    # output2[0:3,1]=2
    # output2[0,3]=1
    # output2[0:2,7]=1
    # output2[1:3,5]=2
    # output2[3,:]=5
    # output2[4:6,1]=2
    # output2[4:9,6]=2
    # output2[4:6,9]=2
    # output2[6:10,4]=1
    # output2[8:10,0]=2
    # output2[8:10,8]=1

    # output4=np.zeros((6,3),dtype=np.uint8)
    # output4[0,0]=2
    # output4[0,1]=9
    # output4[0:2,2]=2
    # output4[1,0]=8
    # output4[1,1]=5
    # output4[2:4,0:2]=2
    # output4[2:4,2]=8
    # output4[4,0]=8
    # output4[4,1]=5
    # output4[4:6,2]=2
    # output4[5,0]=2
    # output4[5,1]=9

    output4 = np.array([[2, 9, 2], [8, 5, 2], [2, 2, 8], [2, 2, 8], [8, 5, 2], [2, 9, 2]], dtype=np.uint8)

    predicted_output_image = output4
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
