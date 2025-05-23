from typing import List
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


def get_prompt(instruction: str, history: List[str]) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and straightforward way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question. "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt

# prompt = """What is the name of the capital city of India, she asked.
#           Please only respond with the city name and then stop talking.
#           He answered:"""


history = []

question = "Which city is the capital of India?"

answer = ""
for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "Which city is of the United States?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
