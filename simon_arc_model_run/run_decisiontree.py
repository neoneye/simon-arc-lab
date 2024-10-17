import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_model.decision_tree_util import DecisionTreeUtil

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

test_index = 0

features = []

noise_levels = [95, 90, 85, 80, 75, 70, 65]
number_of_refinements = len(noise_levels)
last_predicted_output = None
for refinement_index in range(number_of_refinements):
    noise_level = noise_levels[refinement_index]
    print(f"Refinement {refinement_index+1}/{number_of_refinements} noise_level={noise_level}")

    predicted_output = DecisionTreeUtil.predict_output(
        task, 
        test_index, 
        last_predicted_output, 
        refinement_index, 
        noise_level,
        features
    )

    input_image = task.test_input(test_index)
    expected_output_image = task.test_output(test_index)

    show_prediction_result(input_image, predicted_output, expected_output_image, title = task.metadata_task_id)

    last_predicted_output = predicted_output.copy()

