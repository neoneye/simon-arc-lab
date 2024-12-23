import os
from dotenv import dotenv_values
from llama_index.core.llms import ChatMessage
from llama_index.llms.together import TogetherLLM

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
dotenv_dict = dotenv_values(dotenv_path=dotenv_path)

llm = TogetherLLM(
    model="Qwen/QwQ-32B-Preview", 
    api_key=dotenv_dict['TOGETHER_API_KEY']
)

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")
