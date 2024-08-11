# Simon ARC Lab

My experiments with [ARC-AGI](https://github.com/fchollet/ARC-AGI).

---

# Explaining what is inside the datasets that I'm using for LLM training

My datasets are public on huggingface.
https://huggingface.co/neoneye

My dataset generators are public on github (this repo).
https://github.com/neoneye/simon-arc-lab

I use RLE compression, so it may be a bit hard to decipher what is going on.

Below, is the job to identify the colors that are present in a histogram, by eliminating the color counters.

```json
{
    "instruction": "SIMONS-HISTOGRAM, unique colors", 
    "input": "0:3626,2:3280,3:2819,8:677", 
    "output": "0,2,3,8", 
    "benchmark": "dataset=histogram_one group=unique_colors histogram_size=e"
}
```

Below, is the job to subtract 2 histograms, and return what colors and color counters that are remaining.

```json
{
    "instruction": "simons-Arc-Histogram, remove histogram b colors from histogram a", 
    "input": "6:1549,7:1428,2:1325,5:1166,8:1120,0:926,1:734,3:733,9:633\n4:1524,6:97", 
    "output": "7:1428,2:1325,5:1166,8:1120,0:926,1:734,3:733,9:633", 
    "benchmark": "dataset=histogram_two group=a_remove_b_colors histogram_size=e"
}
```

Below, is the job to recognize what cellular automata transformations is happening between input/output images.

```json
{
    "instruction": "SimonCellularAutomata, Recognize the transformation. gameoflife_nowrap,gameoflife_wrap,serviettes_wrap,maze_wrap", 
    "input": "12 21 b262d6a2,a2h62,2626b2d6,b6f2a6,a6h26,b6g26,c6g2,2c6b2b62,a2b6b2b62,a2b6a2a6262,62b6a26b26,c6f26,6j2,i262,d26b2a62,b2g62,b2c62a6a2,b2b6b26a2,a2b6c2a62,i262,h26a2\n12 21 d6d2a6,a6d2b626,6262b6d2,6a2f6a2,j62,a62g62,2a62g6,62a62c62a6,a6262b62626,a6262b62b6,26262e62,c2f62,6,,h6a26,c6f26,b62a626a2a6,b6262b62a6,b6a2f6,6,", 
    "output": "gameoflife_nowrap=0,gameoflife_wrap=0,serviettes_wrap=1,maze_wrap=0", 
    "benchmark": "dataset=cellular_automaton group=recognize_transformation ca_step=1 image_width=medium image_height=large"
}
```

The `"benchmark"` field is not used when training the LLM. I use it afterwards to identify areas where the model fails to predict the correct output.

---

# Simple - without optional dependencies

### Install dependencies

```bash
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
```

### Run tests

```bash
(venv) PROMPT> sh test.sh
```

---

# Advanced - with the optional dependencies

#### Install dependencies

Depends on HuggingFace `transformers`, so it's quite a lot of packages.

```bash
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
(venv) PROMPT> pip install -r requirements_simon_arc_model.txt
```

Now it's possible to run the LLM.

```bash
(venv) PROMPT> python simon_arc_model_run/run_arcagi.py
```

## Generate synthetic datasets

#### Generating `data_rle.jsonl`

```bash
(venv) PROMPT> python dataset_rle.py 
Generated 100 samples, saved to data_rle.jsonl, file size: 11404 bytes.
```

#### Generating `data_image.jsonl`

```bash
(venv) PROMPT> python dataset_image.py
Generated 100 samples, saved to data_image.jsonl, file size: 59621 bytes.
```

#### Generating `data_task.jsonl`

```bash
(venv) PROMPT> python dataset_task.py 
Generated 100 samples, saved to data_task.jsonl, file size: 41954 bytes.
```

# Python package management notes

### Take snapshot of requirements.txt

```
(venv) PROMPT> pip freeze > requirements.txt
```

