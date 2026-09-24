import gradio as gr

from rag_knowledge_base_exploration.answer import answer_question


def to_text(content):
    if isinstance(content, str):
        return content

    text = ""
    for part in content:
        if isinstance(part, dict) and "text" in part:
            text = text + part["text"]
    return text


def chat(message, history):
    our_history = []
    for turn in history:
        our_history.append((turn["role"], to_text(turn["content"])))
    return answer_question(message, our_history)


def main():
    demo = gr.ChatInterface(fn=chat, title="BrightDesk Assistant")
    demo.launch()


if __name__ == "__main__":
    main()