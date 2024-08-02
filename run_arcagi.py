from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from model.runner import *

model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model122'


filename = 'testdata/25ff71a9.json'
task = Task.load_arcagi1(filename)
print(task)

task_without_test_output = task.clone()
task_without_test_output.set_all_test_outputs_to_none()
# print(task_without_test_output.to_arcagi1_json(compact=True))

task_formatter = TaskFormatterRLECompact(task_without_test_output)
output_ids = task_formatter.output_ids()

dataset_name = 'SIMONSOLVETRANSLATE'

dataset_items = []
for test_index in range(task.count_tests):
    # print(f"Test {test_index}")

    input = task_formatter.to_string()

    expected_output = task.test_output(test_index)

    test_output_id = output_ids[task_without_test_output.count_examples + test_index]

    # TODO: let the model make a guess about the output_height
    output_height = 3
    for output_y in range(output_height):
        instruction = f"{dataset_name}, {test_output_id}, predict row {output_y}"

        pixel_list = image_get_row_as_list(expected_output, output_y)

        output = ''.join(map(str, pixel_list))
        benchmark_id = f'dataset={filename} test_index={test_index} output_y={output_y}'

        dataset_item = {
            'instruction': instruction,
            'input': input,
            'output': output,
            'benchmark': benchmark_id
        }
        # print(dataset_item)
        dataset_items.append(dataset_item)

# Initialize runner
runner = Runner(model_directory)
runner.run(dataset_items)
