from datetime import datetime
import os
import sys
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator_score import *
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_multiple_images
from simon_arc_model.arc_bad_prediction import *

max_number_of_bad_predictions_per_task = 10

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    # ('ARC-AGI', 'arcagi', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data')),
    ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('ARC-AGI', 'arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    ('arc-dataset-tama', 'tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    ('Mini-ARC', 'miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    ('ConceptARC', 'conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('ARC-AGI', 'testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

arc_bad_prediction_file = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
arc_bad_prediction_dataset = ARCBadPredictionDataset.load(arc_bad_prediction_file)
# arc_bad_prediction_dataset.display_sample_records()

incorrect_predictions_jsonl_path = arc_bad_prediction_file
#incorrect_predictions_jsonl_path = None

task_ids_with_circle_spirals_list = [
    '08573cc6',
    'e5c44e8f',
    '5c2c9af4',
    'f8c80d96',
    '28e73c20',
]
task_ids_with_circle_spirals = set(task_ids_with_circle_spirals_list)


number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)

    # Keep only that are present in the arc-bad-prediction dataset. Remove the rest.
    task_ids_to_ignore = set()
    for task in taskset.tasks:
        task_id = task.metadata_task_id
        if arc_bad_prediction_dataset.has_records_for_task(dataset_id, task_id):
            continue
        task_ids_to_ignore.add(task_id)
    taskset.remove_tasks_by_id(task_ids_to_ignore, verbose=False)

    # taskset.keep_tasks_with_id(task_ids_with_circle_spirals, verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    gallery_records = []

    pbar = tqdm(taskset.tasks, desc=f"Processing tasks in {groupname}", dynamic_ncols=True)
    for task in pbar:
        task_id = task.metadata_task_id
        pbar.set_postfix_str(f"Task: {task_id}")

        record_list = []
        for test_index in range(task.count_tests):
            records_for_test_index = arc_bad_prediction_dataset.find_records_for_task(dataset_id, task_id, test_index)

            # truncate to the first N bad predictions
            records_for_test_index = records_for_test_index[:max_number_of_bad_predictions_per_task]
            record_list.extend(records_for_test_index)

        ts = TaskSimilarity.create_with_task(task)
        for record in record_list:
            test_index = record.test_index
            # print(f"Task: {task_id} test index: {test_index}")
            predicted_output = record.predicted_output
            # print(predicted_output)
            score = ts.measure_test_prediction(predicted_output, test_index)
            unique_id = str(record.line_number)
            # print(f"Task: {task_id} test index: {test_index} score: {score} line: {unique_id}")

            filename = f'{task_id}_test{test_index}_row{unique_id}_score{score}.png'
            save_path = os.path.join(save_dir, filename)
            input_image = task.test_input(test_index)
            output_image = task.test_output(test_index)
            title_image_list = [
                ('arc', 'input', input_image),
                ('arc', 'predicted', predicted_output),
                ('arc', 'output', output_image),
            ]
            title = f"{dataset_id} {task_id} {test_index} row{unique_id} score{score}"
            show_multiple_images(title_image_list, title=title, save_path=save_path)

            group = f'{dataset_id} {task_id} {test_index}'
            gallery_record = GalleryRecordWithScore(
                group=group,
                image_file_path=save_path, 
                score=score,
                row_id=unique_id,
            )
            gallery_records.append(gallery_record)

    gallery_records.sort(key=lambda x: x.score, reverse=True)
    # print(gallery_records)

    gallery_title = f'{groupname}, {run_id}'
    gallery_generator_score_run(gallery_records, save_dir, title=f"{groupname}, {run_id}")
