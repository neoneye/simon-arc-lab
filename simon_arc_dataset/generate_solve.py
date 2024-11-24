import random
from simon_arc_lab.image_util import *
from simon_arc_lab.image_noise import *
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_distort import *
from simon_arc_lab.image_string_representation import image_to_string

def generate_dataset_item_for_output_image_rle_input_rle_output(
    seed: int, 
    dataset_names: list[str], 
    dataset_id: str, 
    task: Task, 
    test_index: int, 
    transformation_id: str
) -> dict:
    random.seed(seed)
    dataset_name = random.choice(dataset_names)

    output_image = task.test_output(test_index)

    task_without_test_output = task.clone()
    task_without_test_output.set_all_test_outputs_to_none()
    task_formatter = TaskFormatterRLECompact(task_without_test_output)

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

    arc_task_string = task.to_arcagi1_json(True)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={dataset_id} group={transformation_id} predict=image earlier_prediction=none image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    earlier_output_none = ''
    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'arc_task': arc_task_string,
        'test_index': test_index,
        'earlier_output': earlier_output_none,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_for_output_image_with_earlier_prediction_rle_input_rle_output(
    seed: int, 
    dataset_names: list[str], 
    dataset_id: str, 
    task: Task, 
    test_index: int, 
    earlier_output_image: np.array, 
    transformation_id: str, 
    benchmark_earlier_prediction_id: str
) -> dict:
    random.seed(seed)
    dataset_name = random.choice(dataset_names)

    output_image = task.test_output(test_index)

    t2 = task.clone()
    t2.set_all_test_outputs_to_none()
    image_index = t2.count_examples + test_index
    t2.output_images[image_index] = earlier_output_image
    t2.metadata_task_id = f'{t2.metadata_task_id} {benchmark_earlier_prediction_id}'
    # t2.show()

    task_formatter = TaskFormatterRLECompact(t2)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[t2.count_examples + test_index]

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

    arc_task_string = task.to_arcagi1_json(True)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={dataset_id} group={transformation_id} predict=image earlier_prediction={benchmark_earlier_prediction_id} image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    serializable_earlier_output_image = earlier_output_image.tolist()
    earlier_output = json.dumps(serializable_earlier_output_image, separators=(',', ':'))

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'arc_task': arc_task_string,
        'test_index': test_index,
        'earlier_output': earlier_output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_for_output_image_rle_input_rawpixel_output(
    seed: int, 
    dataset_names: list[str], 
    dataset_id: str, 
    task: Task, 
    test_index: int, 
    transformation_id: str
) -> dict:
    random.seed(seed)
    dataset_name = random.choice(dataset_names)

    output_image = task.test_output(test_index)

    task_without_test_output = task.clone()
    task_without_test_output.set_all_test_outputs_to_none()
    task_formatter = TaskFormatterRLECompact(task_without_test_output)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[task.count_examples + test_index]

    instructions = [
        f"{dataset_name}, {test_output_id}, predict raw pixels",
        f"{dataset_name} '{test_output_id}' predict the raw pixels",
        f"{dataset_name}, '{test_output_id}', predict the raw pixels",
        f"{dataset_name} predict raw pixels for {test_output_id}",
        f"{dataset_name} predict raw pixels for '{test_output_id}'",
        f"{dataset_name}, {test_output_id}, predict pixels",
        f"{dataset_name} '{test_output_id}' predict the pixels",
        f"{dataset_name}, '{test_output_id}', predict the pixels",
        f"{dataset_name} predict pixels for {test_output_id}",
        f"{dataset_name} predict pixels for '{test_output_id}'",
    ]
    instruction = random.choice(instructions)

    input = task_formatter.to_string()
    # print(input)

    output = image_to_string(output_image)

    arc_task_string = task.to_arcagi1_json(True)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={dataset_id} group={transformation_id} predict=rawpixels earlier_prediction=none image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    earlier_output_none = ''
    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'arc_task': arc_task_string,
        'test_index': test_index,
        'earlier_output': earlier_output_none,
        'benchmark': benchmark_id
    }
    return result_dict

class DatasetItemListBuilder:
    def __init__(self, seed: int, task: Task, dataset_names: list[str], dataset_id: str, transformation_id: str):
        self.seed = seed
        self.task = task
        self.dataset_names = dataset_names
        self.dataset_id = dataset_id
        self.transformation_id = transformation_id
        self.accumulated_dataset_items = []

    def append_arcagi1_json(self):
        """
        Export the entire Task to the original ARC-AGI version 1 JSON representation.

        This adds a 'metadata' field with info about how the Task was generated.
        """
        task_dict = self.task.to_arcagi1_dict()
        metadata = {}
        if self.task.metadata_task_id is not None:
            if len(self.task.metadata_task_id) > 0:
                if self.task.metadata_task_id != self.transformation_id:
                    metadata['task_id'] = self.task.metadata_task_id
        
        metadata['dataset_id'] = self.dataset_id
        metadata['transformation_id'] = self.transformation_id
        
        if len(metadata) > 0:
            task_dict['metadata'] = metadata
        self.accumulated_dataset_items.append(task_dict)

    def append_image_rle_output(self):
        """
        Predict the entire output image with RLE compression.
        """
        for test_index in range(self.task.count_tests):
            dataset_item = generate_dataset_item_for_output_image_rle_input_rle_output(
                self.seed + test_index * 100 + 2000, 
                self.dataset_names, 
                self.dataset_id,
                self.task, 
                test_index, 
                self.transformation_id
            )
            self.accumulated_dataset_items.append(dataset_item)

    def append_image_rawpixel_output(self):
        """
        Predict the entire output image without RLE compression.
        """
        for test_index in range(self.task.count_tests):
            dataset_item = generate_dataset_item_for_output_image_rle_input_rawpixel_output(
                self.seed + test_index * 100 + 2000, 
                self.dataset_names, 
                self.dataset_id,
                self.task, 
                test_index, 
                self.transformation_id
            )
            self.accumulated_dataset_items.append(dataset_item)

    def append_image_with_earlier_prediction_very_close_to_expected_output(self):
        """
        Predict the entire output image, with help from an earlier prediction that is very close to the expected output.
        """
        for test_index in range(self.task.count_tests):
            output_image = self.task.test_output(test_index)
            earlier_predicted_image = image_distort(output_image, 1, 10, self.seed + test_index * 100 + 1000)

            dataset_item = generate_dataset_item_for_output_image_with_earlier_prediction_rle_input_rle_output(
                self.seed + test_index * 100 + 2000, 
                self.dataset_names, 
                self.dataset_id,
                self.task, 
                test_index, 
                earlier_predicted_image,
                self.transformation_id,
                'output_distort_1steps_10percent'
            )
            self.accumulated_dataset_items.append(dataset_item)

    def append_image_with_earlier_prediction_similar_to_original_input(self):
        """
        Predict the entire output image.
        With a distorted input image. No help from the expected output image.
        """
        for test_index in range(self.task.count_tests):
            input_image = self.task.test_input(test_index)
            earlier_predicted_image = image_distort(input_image, 1, 10, self.seed + test_index * 100 + 1000)

            dataset_item = generate_dataset_item_for_output_image_with_earlier_prediction_rle_input_rle_output(
                self.seed + test_index * 100 + 2000, 
                self.dataset_names, 
                self.dataset_id,
                self.task, 
                test_index, 
                earlier_predicted_image,
                self.transformation_id,
                'input_distort_1steps_10percent'
            )
            self.accumulated_dataset_items.append(dataset_item)

    def append_image_with_earlier_prediction_original_output_with_1_bad_pixel(self):
        """
        Repair 1 bad pixel.
        With the expected output image, but with one bad pixel.
        """
        for test_index in range(self.task.count_tests):
            iteration_seed = self.seed + test_index * 100 + 1000 + 38383231
            output_image = self.task.test_output(test_index)
            earlier_predicted_image = image_noise_one_pixel(output_image, iteration_seed + 1)

            dataset_item = generate_dataset_item_for_output_image_with_earlier_prediction_rle_input_rle_output(
                iteration_seed + 4, 
                self.dataset_names, 
                self.dataset_id,
                self.task, 
                test_index, 
                earlier_predicted_image,
                self.transformation_id,
                'repair_1_bad_pixel'
            )
            self.accumulated_dataset_items.append(dataset_item)

    def append_image_randomized(self):
        j = self.seed % 4
        if j == 0:
            self.append_image_rle_output()
        elif j == 1:
            self.append_image_with_earlier_prediction_very_close_to_expected_output()
        elif j == 2:
            self.append_image_with_earlier_prediction_original_output_with_1_bad_pixel()
        else:
            self.append_image_with_earlier_prediction_similar_to_original_input()

    def dataset_items(self):
        return self.accumulated_dataset_items
