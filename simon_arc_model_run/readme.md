# Simon ARC Model Run

Command line interface.

## Run the solver that uses decisiontrees

Identify what ARC-AGI puzzles that are predicted correct/incorrect. Creates a dir, where each prediction can be inspected.

```bash
(venv) PROMPT> python simon_arc_model_run/run_tasks_with_decisiontree.py 
Run id: 20241112_144837
Using WorkManager of type: WorkManagerDecisionTree
Number of task ids to ignore: 93
Processing 1 of 1. Group name 'arcagi_training'. Results will be saved to 'run_tasks_result/20241112_144837/arcagi_training'
Loading 405 tasks from /Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training
Saving images to directory: run_tasks_result/20241112_144837/arcagi_training
Processing work items: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:31<00:00,  1.57s/it, correct=0]
Removed 0 work items where the input and output is identical. Remaining are 20 work items.
Number of correct solutions: 0
None_INCORRECT: 20
```


## Run the solver that uses LLM

Identify what ARC-AGI puzzles that are predicted correct/incorrect. Creates a dir, where each prediction can be inspected.

```bash
(venv) PROMPT> python simon_arc_model_run/run_tasks_with_llm.py 
Using WorkManager of type: WorkManagerSimple
context length: 512, max prompt length: 500
Number of task ids to ignore: 93
Processing 1 of 2. Group name 'arcagi_training'. Results will be saved to 'run_tasks_result/682/arcagi_training'
Loading 405 tasks from /Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training
Removed 268 work items with too long prompt. Remaining are 576 work items.
Saving images to directory: run_tasks_result/682/arcagi_training
Processing work items:   1%|█▏                          | 7/576 [00:10<10:38,  1.12s/it, correct=0]
```


## Run benchmark

This measures accuracy of the model. 

Beforehand create a novel dataset jsonl file with a random seed, that the model hasn't previously been trained on.

This way the task is to make predictions on new data, it has never been trained on.

```bash
(venv) PROMPT> python simon_arc_model_run/run_benchmark.py
Processing entries:   0%|██████████████████████████████           | 9/100000 [00:13<33:19:26,  1.20s/it]Correct sum: 9
dataset=solve_translate group=translate_xminus1 predict=image image_width=small image_height=small task_pixels=a: 1
dataset=solve_translate group=translate_xminus1minus1 predict=image image_width=small image_height=small task_pixels=a: 1
dataset=solve_translate group=translate_xminus1plus1 predict=image image_width=small image_height=small task_pixels=a: 1
dataset=solve_translate group=translate_xplus1 predict=image image_width=small image_height=small task_pixels=a: 1
dataset=solve_translate group=translate_xplus1yminus1 predict=image image_width=small image_height=small task_pixels=a: 1
dataset=solve_translate group=translate_xplus1yminus1 predict=image image_width=small image_height=small task_pixels=b: 1
dataset=solve_translate group=translate_xplus1yplus1 predict=image image_width=small image_height=small task_pixels=b: 1
dataset=solve_translate group=translate_yminus1 predict=image image_width=small image_height=small task_pixels=a: 1
dataset=solve_translate group=translate_yplus1 predict=image image_width=small image_height=small task_pixels=a: 1

Incorrect sum: 1
dataset=solve_translate group=translate_xminus1 predict=image image_width=small image_height=small task_pixels=b: 1
CTRL-C to abort
```

## Run tasks

Export the predicted outputs as images.

```bash
(venv) PROMPT> python simon_arc_model_run/run_tasks.py
Loading 405 tasks from /absolute/path/to/arc-dataset-collection/dataset/ARC/data/training
Removed 130 work items with too long prompt. Remaining are 292 work items.
Saving images to directory: run_tasks_result
Processing work items:   3%|██▋       | 8/292 [00:08<06:26,  1.36s/it, correct=0]
CTRL-C to abort
```
