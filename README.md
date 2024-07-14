# Simon ARC Lab

My experiments with [ARC-AGI](https://github.com/fchollet/ARC-AGI).

### Install dependencies

```
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
```

### Take snapshot of requirements.txt

```
(venv) PROMPT> pip freeze > requirements.txt
```

### Run tests

```
(venv) PROMPT> sh test.sh
```

---

## Generate synthetic datasets

#### Generating `data_rle.jsonl`

```
(venv) PROMPT> python generate_dataset_rle.py 
Generated 100 samples, saved to data_rle.jsonl, file size: 11404 bytes.
```

#### Generating `data_image.jsonl`

```
(venv) PROMPT> python generate_dataset_image.py
Generated 100 samples, saved to data_image.jsonl, file size: 59621 bytes.
```

#### Generating `data_task.jsonl`

```
(venv) PROMPT> python generate_dataset_task.py 
Generated 100 samples, saved to data_task.jsonl, file size: 41954 bytes.
```
