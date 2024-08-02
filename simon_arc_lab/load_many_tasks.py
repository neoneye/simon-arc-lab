import os
from .task import Task

def load_many_tasks(path_to_task_dir: str) -> list[Task]:
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

        # print("Loading task", task_id)
        task = Task.load_arcagi1(path)
        task.task_id = task_id
        tasks.append(task)
    print(f"Loading {len(json_paths_sorted)} tasks from {path_to_task_dir}")
    return tasks

if __name__ == '__main__':
    path = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training'
    tasks = load_many_tasks(path)
