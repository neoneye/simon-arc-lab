from .rle.serialize import serialize
import json

class TaskFormatter:
    def __init__(self):
        self.input_images = []
        self.output_images = []
        self.count_examples = 0
        self.count_tests = 0

    def append_pair(self, input_image, output_image, is_example):
        self.assert_count()
        self.input_images.append(input_image)
        self.output_images.append(output_image)
        if is_example:
            self.count_examples += 1
        else:
            self.count_tests += 1
        self.assert_count()

    def count(self):
        self.assert_count()
        return len(self.input_images)

    def assert_count(self):
        assert len(self.input_images) == len(self.output_images)
        assert self.count_examples + self.count_tests == len(self.input_images)

    def input_ids(self):
        self.assert_count()
        names = []
        for i in range(len(self.input_images)):
            if i < self.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Input {i} {name}")
        return names

    def output_ids(self):
        self.assert_count()
        names = []
        for i in range(len(self.input_images)):
            if i < self.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Output {i} {name}")
        return names
    
    def pair_ids(self):
        self.assert_count()
        names = []
        for i in range(len(self.input_images)):
            if i < self.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Pair {i} {name}")
        return names
    
    def max_image_size(self):
        """
        Find (width, height) of the biggest images in the task.

        Where the Output is None, the image size is not included.
        """
        self.assert_count()
        width = 0
        height = 0
        for i in range(len(self.input_images)):
            width = max(width, self.input_images[i].shape[1])
            height = max(height, self.input_images[i].shape[0])
            if self.output_images[i] is not None:
                width = max(width, self.output_images[i].shape[1])
                height = max(height, self.output_images[i].shape[0])
        return (width, height)
    
    def total_pixel_count(self) -> int:
        """
        Count the total number of pixels in all images.

        Where the Output is None, the pixel count is not included.
        """
        self.assert_count()
        count = 0
        for i in range(len(self.input_images)):
            count += self.input_images[i].shape[0] * self.input_images[i].shape[1]
            if self.output_images[i] is not None:
                count += self.output_images[i].shape[0] * self.output_images[i].shape[1]
        return count
    
    def serialize_input_image(self, i: int) -> str:
        self.assert_count()
        if i < 0 or i >= len(self.input_images):
            raise ValueError("Invalid index")
        return serialize(self.input_images[i])
    
    def serialize_output_image(self, i: int) -> str:
        self.assert_count()
        if i < 0 or i >= len(self.output_images):
            raise ValueError("Invalid index")
        output_image = self.output_images[i]
        if output_image is None:
            return "None"
        else:
            return serialize(output_image)

    def to_string(self) -> str:
        self.assert_count()
        input_ids = self.input_ids()
        output_ids = self.output_ids()
        s = ""
        for i in range(len(self.input_images)):
            if i > 0:
                s += "\n"
            s += input_ids[i] + "\n"
            s += self.serialize_input_image(i) + "\n"
            s += output_ids[i] + "\n"
            s += self.serialize_output_image(i)
        return s

    def to_json(self) -> str:
        array_train = []
        array_test = []
        for i in range(self.count()):
            dict = {
                'input': self.input_images[i].tolist(),
                'output': self.output_images[i].tolist(),
            }
            if i < self.count_examples:
                array_train.append(dict)
            else:
                array_test.append(dict)

        dict = {
            'train': array_train,
            'test': array_test,
        }
        return json.dumps(dict)
