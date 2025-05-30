import chainlit as cl
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    global llm
    if message.content.lower() in ["use llama2", "use orca"]:
        model_name = message.content.lower().split()[1]
        if model_name == "llama2":
            llm = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q8_0.gguf"
            )
            await cl.Message(content="Model changed to Llama").send()
        elif model_name == "orca":
            llm = AutoModelForCausalLM.from_pretrained(
                "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
            )
            await cl.Message(content="Model changed to Orca").send()
        else:
            await cl.Message(content="Model not found, keeping old model").send()
        return
    if message.content.lower() == "forget everything":
        cl.user_session.set("message_history", [])
        await cl.Message(content="Uh oh, I've just forgotten our conversation history").send()
        return
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()
    print(message.content)
    # ...existing code...
    if message.content == "forget everything":
        message_history.clear()
        await cl.Message(content="Uh oh, I've just forgotten our conversation history").send()
        return
# ...existing code...

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm

    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    await cl.Message("Model initialized. How can I help you?").send()
