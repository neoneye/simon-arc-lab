# Experiments with LlamaIndex

[LlamaIndex repo](https://github.com/run-llama/llama_index)

- Free solution: Ollama can run models on localhost. I prefer this one, however my computer cannot handle big models.
- Paid solution: OpenAI.
- Paid solution: [together.ai](https://www.together.ai/), can run many of the open source models.

## Config

Copy the `.env.example` to a file named `.env`.

Replace the `OPENAI_API_KEY` placeholder with your own key.

Replace the `TOGETHER_API_KEY` placeholder with your own key.

## Install

Files in this dir makes use of some python packages, that the other parts of the project does not make use of. I haven't included them in the global `requirements.txt`.

### Required packages

```bash
PROMPT> pip install python-dotenv
PROMPT> pip install llama-index
```

### Optional packages

```bash
PROMPT> pip install llama-index-llms-ollama
PROMPT> pip install llama-index-llms-openai
PROMPT> pip install llama-index-llms-together
```

## Usage

```bash
PROMPT> python simon_llamaindex/run_chat_openai.py 
Ahoy there, matey! Ye can call me Captain Chatbeard, the most colorful pirate to sail the digital seas! What treasure of knowledge be ye seekin' today? Arrr! 🏴‍☠️✨%
```
