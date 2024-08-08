# Simon ARC Lab

My experiments with [ARC-AGI](https://github.com/fchollet/ARC-AGI).

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

