from .rle.serialize import serialize
from .task import Task

class TaskFormatterRLE:
    def __init__(self, task: Task):
        self.task = task
    
    def input_ids(self) -> list[str]:
        self.task.assert_count()
        names = []
        for i in range(self.task.count()):
            if i < self.task.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Input {i} {name}")
        return names

    def output_ids(self) -> list[str]:
        self.task.assert_count()
        names = []
        for i in range(self.task.count()):
            if i < self.task.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Output {i} {name}")
        return names
    
    def pair_ids(self) -> list[str]:
        self.task.assert_count()
        names = []
        for i in range(self.task.count()):
            if i < self.task.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Pair {i} {name}")
        return names
    
    def serialize_input_image(self, i: int) -> str:
        self.task.assert_count()
        if i < 0 or i >= self.task.count():
            raise ValueError("Invalid index")
        return serialize(self.task.input_images[i])
    
    def serialize_output_image(self, i: int) -> str:
        self.task.assert_count()
        if i < 0 or i >= self.task.count():
            raise ValueError("Invalid index")
        output_image = self.task.output_images[i]
        if output_image is None:
            return "None"
        else:
            return serialize(output_image)

    def to_string(self) -> str:
        self.task.assert_count()
        input_ids = self.input_ids()
        output_ids = self.output_ids()
        parts = []
        for i in range(self.task.count()):
            parts.append(input_ids[i])
            parts.append(self.serialize_input_image(i))
            parts.append(output_ids[i])
            parts.append(self.serialize_output_image(i))
        s = "\n".join(parts)
        return s
