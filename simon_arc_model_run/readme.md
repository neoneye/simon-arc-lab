# Simon ARC Model Run

Command line interface.

## The decisiontree solver

In the ARC Prize 2024 contest, my decisiontree solver got `score=1`. It solved 1 of the 100 hidden puzzles.

The decisiontree solver is better than the llm solver.

This code creates a dir, where each prediction can be inspected.

Here is a visualization of the solved puzzles:
- [ARC-AGI training](https://neoneye.github.io/simon-arc-lab-web/model/2024-oct-17-1318/arcagi_training/)
- [ARC-AGI evaluation](https://neoneye.github.io/simon-arc-lab-web/model/2024-oct-17-1318/arcagi_evaluation/)

To run the solver:
```bash
python simon_arc_model_run/run_tasks_with_decisiontree.py
Run id: 20241201_121749
Using WorkManager of type: WorkManagerDecisionTree
Number of task ids to ignore: 93
Processing 1 of 1. Group name 'arcagi_training'. Results will be saved to 'run_tasks_result/20241201_121749/arcagi_training'
Loading 405 tasks from /Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training
Saving images to directory: run_tasks_result/20241201_121749/arcagi_training
Processing work items: 100%|████████████████████████████████████████████████████████| 273/273 [00:33<00:00,  8.25it/s, correct=39]
Removed 4 work items where the input and output is identical. Remaining are 269 work items.
Number of correct solutions: 39
None_INCORRECT: 229
None_CORRECT: 40
```

## The LLM solver

In the ARC Prize 2024 contest, my llm solver got `score=1`. It solved 1 of the 100 hidden puzzles.

It's based on the `CodeT5` llm. The prompt and response use RLE compression. The llm is worse than the decisiontree solver.

This code creates a dir, where each prediction can be inspected.

Here is a visualization of the solved puzzles:
- [ARC-AGI training](https://neoneye.github.io/simon-arc-lab-web/model/625/arcagi_training/)
- [ARC-AGI evaluation](https://neoneye.github.io/simon-arc-lab-web/model/625/arcagi_evaluation/)

To run the solver:
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
