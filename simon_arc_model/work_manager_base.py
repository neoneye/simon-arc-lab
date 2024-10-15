from typing import Optional

class WorkManagerBase:
    def truncate_work_items(self, max_count: int):
        raise NotImplementedError()

    def discard_items_with_too_long_prompts(self, max_prompt_length: int):
        raise NotImplementedError()

    def discard_items_with_too_short_prompts(self, min_prompt_length: int):
        raise NotImplementedError()

    def discard_items_where_predicted_output_is_identical_to_the_input(self):
        raise NotImplementedError()
    
    def process_all_work_items(self, show: bool = False, save_dir: Optional[str] = None):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def save_arcprize2024_submission_file(self, path_to_json_file: str):
        raise NotImplementedError()
