# Simon ARC Model Run

Command line interface.

## Run ARC-AGI

Identify what ARC-AGI puzzles that are predicted correct/incorrect.

```bash
(venv) PROMPT> python simon_arc_model_run/run_arcagi.py
Loading 8 tasks from /Users/neoneye/git/simon_arc_lab/testdata
Generated 9 dataset items
Processing entries: 100%|██████████████████████████████████████████████████| 9/9 [00:04<00:00,  1.88it/s]

Final Summary
Correct sum: 7
dataset=25ff71a9 predict=image test_index=0: 1
dataset=3c9b0459 predict=image test_index=0: 1
dataset=5582e5ca predict=image test_index=0: 1
dataset=6150a2bd predict=image test_index=0: 1
dataset=68b16354 predict=image test_index=0: 1
dataset=74dd1130 predict=image test_index=0: 1
dataset=Most_Common_color_l6ab0lf3xztbyxsu3p predict=image test_index=0: 1

Incorrect sum: 2
dataset=25ff71a9 predict=image test_index=1: 1
dataset=ed36ccf7 predict=image test_index=0: 1
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
