import random
from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.benchmark import *

def generate_dataset_item_for_pixels_in_output_row(seed: int, dataset_names: list[str], benchmark_dataset_name: str, task: Task, test_index: int, test_output_y: int, pixel_list: list[int], transformation_id: str) -> dict:
    random.seed(seed)
    dataset_name = random.choice(dataset_names)

    task_formatter = TaskFormatterRLECompact(task)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[task.count_examples + test_index]

    instructions = [
        f"{dataset_name}, {test_output_id}, predict row {test_output_y}",
        f"{dataset_name} '{test_output_id}' predict row {test_output_y}",
        f"{dataset_name} '{test_output_id}' predict the row {test_output_y}",
        f"{dataset_name}, '{test_output_id}', predict the row {test_output_y}",
        f"{dataset_name}, '{test_output_id}', predict y={test_output_y}",
        f"{dataset_name} {test_output_id} predict y={test_output_y}",
        f"{dataset_name} predict y={test_output_y} for {test_output_id}",
        f"{dataset_name} predict row {test_output_y} for {test_output_id}",
    ]
    instruction = random.choice(instructions)

    input = task_formatter.to_string()
    # print(input)

    output = ''.join(map(str, pixel_list))

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} predict=pixels image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_for_number_of_output_rows(seed: int, dataset_names: list[str], benchmark_dataset_name: str, task: Task, test_index: int, output_image: np.array, transformation_id: str) -> dict:
    random.seed(seed)
    dataset_name = random.choice(dataset_names)

    task_formatter = TaskFormatterRLECompact(task)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[task.count_examples + test_index]

    instructions = [
        f"{dataset_name}, {test_output_id}, predict row count",
        f"{dataset_name} '{test_output_id}' predict row count",
        f"{dataset_name} '{test_output_id}' predict the row count",
        f"{dataset_name}, '{test_output_id}', predict the row count",
        f"{dataset_name}, '{test_output_id}', predict the height",
        f"{dataset_name}, '{test_output_id}', predict height",
        f"{dataset_name} {test_output_id} predict the height",
        f"{dataset_name} {test_output_id} predict height",
    ]
    instruction = random.choice(instructions)

    input = task_formatter.to_string()
    # print(input)

    output_height = output_image.shape[0]
    output = str(output_height)
    # print(output)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} predict=height image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_for_output_image(seed: int, dataset_names: list[str], benchmark_dataset_name: str, task: Task, test_index: int, output_image: np.array, transformation_id: str) -> dict:
    random.seed(seed)
    dataset_name = random.choice(dataset_names)

    task_formatter = TaskFormatterRLECompact(task)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[task.count_examples + test_index]

    instructions = [
        f"{dataset_name}, {test_output_id}, predict image",
        f"{dataset_name} '{test_output_id}' predict the image",
        f"{dataset_name}, '{test_output_id}', predict the image",
        f"{dataset_name} predict image for {test_output_id}",
        f"{dataset_name} predict image for '{test_output_id}'",
    ]
    instruction = random.choice(instructions)

    input = task_formatter.to_string()
    # print(input)

    output = serialize(output_image)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} predict=image image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

class DatasetItemListBuilder:
    def __init__(self, seed: int, task: Task, dataset_names: list[str], benchmark_dataset_name: str, transformation_id: str):
        self.seed = seed
        self.task = task
        self.dataset_names = dataset_names
        self.benchmark_dataset_name = benchmark_dataset_name
        self.transformation_id = transformation_id

        task_without_test_output = task.clone()
        task_without_test_output.set_all_test_outputs_to_none()
        self.task_without_test_output = task_without_test_output

        self.accumulated_dataset_items = []

    def append_pixels(self):
        random.seed(self.seed * 1000003 + 1)

        # Predict the pixels of the output image
        for test_index in range(self.task.count_tests):
            output_image = self.task.test_output(test_index)
            output_height = output_image.shape[0]
            for output_y in range(output_height):
                pixels = image_get_row_as_list(output_image, output_y)
                dataset_item = generate_dataset_item_for_pixels_in_output_row(
                    self.seed + output_y + test_index * 100, 
                    self.dataset_names, 
                    self.benchmark_dataset_name,
                    self.task_without_test_output, 
                    test_index, 
                    output_y, 
                    pixels,
                    self.transformation_id
                )
                self.accumulated_dataset_items.append(dataset_item)

    def append_height(self):
        random.seed(self.seed * 1000005 + 97)

        # Predict the number of rows in the output image
        for test_index in range(self.task.count_tests):
            output_image = self.task.test_output(test_index)
            dataset_item = generate_dataset_item_for_number_of_output_rows(
                self.seed + test_index * 100 + 1000, 
                self.dataset_names, 
                self.benchmark_dataset_name,
                self.task_without_test_output, 
                test_index, 
                output_image,
                self.transformation_id
            )
            self.accumulated_dataset_items.append(dataset_item)

    def append_image(self):
        random.seed(self.seed * 1000007 + 32)

        # Predict the entire output image
        for test_index in range(self.task.count_tests):
            output_image = self.task.test_output(test_index)
            dataset_item = generate_dataset_item_for_output_image(
                self.seed + test_index * 100 + 2000, 
                self.dataset_names, 
                self.benchmark_dataset_name,
                self.task_without_test_output, 
                test_index, 
                output_image,
                self.transformation_id
            )
            self.accumulated_dataset_items.append(dataset_item)

    def dataset_items(self):
        return self.accumulated_dataset_items
