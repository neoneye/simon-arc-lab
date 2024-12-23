import os
from dotenv import dotenv_values
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
dotenv_dict = dotenv_values(dotenv_path=dotenv_path)

llm = Ollama(model="llama3.1:latest", request_timeout=120.0)

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")
