import os
from .task import Task

def load_tasks_from_directory(path_to_task_dir: str) -> list[Task]:
    tasks = []
    json_paths = []
    for root, dirs, files in os.walk(path_to_task_dir):
        for file in files:
            if file.endswith(".json"):
                json_paths.append(os.path.join(root, file))
    json_paths_sorted = sorted(json_paths)
    for path in json_paths_sorted:
        # Remove path, and remove the file extension
        task_id = os.path.splitext(os.path.basename(path))[0]

        # Absolute path to task file
        absolute_path = os.path.abspath(path)

        # print("Loading task", task_id)
        task = Task.load_arcagi1(path)
        task.metadata_task_id = task_id
        task.metadata_path = absolute_path
        tasks.append(task)
    print(f"Loading {len(json_paths_sorted)} tasks from {path_to_task_dir}")
    return tasks

if __name__ == '__main__':
    # How to run this snippet
    # PROMPT> python -m simon_arc_lab.load_tasks_from_directory

    path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training'
    tasks = load_tasks_from_directory(path)
    for task in tasks[:2]:
        print(task)
        task.show()
