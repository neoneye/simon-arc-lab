# IDEA: Rotate only the input images, don't touch the output images
# IDEA: Rotate only the output images, don't touch the input images
#
# IDEA: Flip only the input images, don't touch the output images
# IDEA: Flip only the output images, don't touch the input images
#
# IDEA: Skew only the input images, don't touch the output images
# IDEA: Skew only the output images, don't touch the input images
#
# IDEA: Shuffle the order of pairs. It may impact the model's ability to solve the task.
#
# IDEA: Tasks with 2 colors, normalize so they use the same color
from .task import Task
from .image_util import *
from .rle.serialize import serialize

class TaskMutatorBase:
    def __init__(self, task: Task):
        raise NotImplementedError()

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError()

    def transformed_task(self) -> Task:
        raise NotImplementedError()

    def reverse_transformation(self, transformed_image: np.array, pair_index: int) -> np.array:
        """
        Return the image to its original format, before the transformation was applied.

        Example: The task gets transposed and the output needs to be transposed back.
        """
        raise NotImplementedError()

class TaskMutatorOriginal(TaskMutatorBase):
    def __init__(self, task: Task):
        self.task = task.clone()

    @classmethod
    def name(cls) -> str:
        return 'original'
    
    def transformed_task(self) -> Task:
        return self.task.clone()
    
    def reverse_transformation(self, transformed_image: np.array, pair_index: int) -> np.array:
        return transformed_image.copy()

# Transpose all images in the task
class TaskMutatorTranspose(TaskMutatorBase):
    def __init__(self, task: Task):
        new_task = task.clone()
        for i in range(task.count()):
            input_image = task.input_images[i]
            output_image = task.output_images[i]
            new_task.input_images[i] = np.transpose(input_image)
            new_task.output_images[i] = np.transpose(output_image)
        self.task = new_task

    @classmethod
    def name(cls) -> str:
        return 'transpose'
    
    def transformed_task(self) -> Task:
        return self.task.clone()
    
    def reverse_transformation(self, transformed_image: np.array, pair_index: int) -> np.array:
        return np.transpose(transformed_image)

# Tasks with different orientation, normalize so they have the same orientation.
# If images compress well in one direction, and worse in another direction, then it may indicate that 
# the pair has a different orientation.
class TaskMutatorTransposeSoInputIsMostCompact(TaskMutatorBase):
    def __init__(self, task: Task):
        new_task = task.clone()
        istransposed_list = []
        for i in range(task.count()):
            input_image = task.input_images[i]
            serialize0 = serialize(input_image)
            serialize1 = serialize(np.transpose(input_image))
            # Shortest RLE representation is the most compact
            istransposed = len(serialize1) < len(serialize0)

            if istransposed:
                new_input_image = np.transpose(input_image)
            else:
                new_input_image = input_image
            
            if istransposed:
                new_output_image = np.transpose(task.output_images[i])
            else:
                new_output_image = task.output_images[i].copy()

            new_task.input_images[i] = new_input_image
            new_task.output_images[i] = new_output_image
            
            istransposed_list.append(istransposed)

        self.istransposed_list = istransposed_list
        self.task = new_task

    @classmethod
    def name(cls) -> str:
        return 'mostcompact'
    
    def transformed_task(self) -> Task:
        return self.task.clone()
    
    def reverse_transformation(self, transformed_image: np.array, pair_index: int) -> np.array:
        istransposed = self.istransposed_list[pair_index]
        if istransposed:
            return np.transpose(transformed_image)
        else:
            return transformed_image.copy()
