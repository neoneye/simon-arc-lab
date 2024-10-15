import numpy as np
from .work_item import WorkItem

class WorkItemList:
    @classmethod
    def discard_items_with_too_long_prompts(cls, work_items: list[WorkItem], max_prompt_length: int) -> list[WorkItem]:
        """
        Ignore those where the prompt longer than what the model can handle.
        """
        count_before = len(work_items)
        filtered_work_items = []
        for work_item in work_items:
            prompt_length = len(work_item.predictor.prompt())
            if prompt_length <= max_prompt_length:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        print(f'Removed {count_before - count_after} work items with too long prompt. Remaining are {count_after} work items.')
        return filtered_work_items

    @classmethod
    def discard_items_with_too_short_prompts(cls, work_items: list[WorkItem], min_prompt_length: int) -> list[WorkItem]:
        """
        Ignore those where the prompt shorter than N tokens.
        """
        count_before = len(work_items)
        filtered_work_items = []
        for work_item in work_items:
            prompt_length = len(work_item.predictor.prompt())
            if prompt_length >= min_prompt_length:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        print(f'Removed {count_before - count_after} work items with too short prompt. Remaining are {count_after} work items.')
        return filtered_work_items

    @classmethod
    def discard_items_where_predicted_output_is_identical_to_the_input(cls, work_items: list[WorkItem]) -> list[WorkItem]:
        """
        Usually in ARC-AGI the predicted output image is supposed to be different from the input image.
        There are ARC like datasets where the input and output may be the same, but it's rare.
        It's likely a mistake when input and output is the same.
        """
        count_before = len(work_items)
        filtered_work_items = []
        for work_item in work_items:
            if work_item.predicted_output_image is None:
                filtered_work_items.append(work_item)
                continue
            input_image = work_item.task.test_input(work_item.test_index)
            predicted_image = work_item.predicted_output_image
            is_identical = np.array_equal(input_image, predicted_image)
            if not is_identical:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        print(f'Removed {count_before - count_after} work items where the input and output is identical. Remaining are {count_after} work items.')
        return filtered_work_items

