# Install dependencies

# Simple - without optional dependencies

```bash
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
```

Run tests to verify that the dependencies have been installed correctly and that things are working.

```bash
(venv) PROMPT> sh test.sh
```

# Advanced - with the optional dependencies

This is when using the LLM model.

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


# Python package management notes

### Take snapshot of requirements.txt

```
(venv) PROMPT> pip freeze > requirements.txt
```

