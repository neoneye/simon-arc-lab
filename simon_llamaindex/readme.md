# Experiments with LlamaIndex

[LlamaIndex repo](https://github.com/run-llama/llama_index)

## Config

Copy the `.env.example` to a file named `.env`.

Replace the placeholders with your own `OPENAI_API_KEY`.

## Install

Files in this dir makes use of some python packages, that the other parts of the project does not make use of. I haven't included them in the global `requirements.txt`.

```bash
PROMPT> pip install python-dotenv
PROMPT> pip install llama-index llama-index-llms-openai
PROMPT> pip install llama-index-llms-ollama
```

## Usage

```bash
PROMPT> python simon_llamaindex/run_chat_openai.py 
Ahoy there, matey! Ye can call me Captain Chatbeard, the most colorful pirate to sail the digital seas! What treasure of knowledge be ye seekin' today? Arrr! 🏴‍☠️✨%
```
