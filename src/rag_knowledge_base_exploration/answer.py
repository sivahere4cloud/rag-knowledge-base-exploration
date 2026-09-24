from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_knowledge_base_exploration.ingest import DB_PATH, EMBED_MODEL

load_dotenv(override=True)

CHAT_MODEL = "gpt-4.1-nano"
TOP_K = 3

SYSTEM_PROMPT = """You are an assistant for BrightDesk IT Solutions.
Answer questions using ONLY the context provided.
If the context contains several people or services that match the question,
briefly describe each one.
If the answer is not in the context at all, say you don't know."""

REWRITE_PROMPT = """Rewrite the user's latest question as a standalone question
that makes sense without the conversation history.
Replace pronouns like 'she', 'he', 'it' or 'that' with what they refer to.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question."""

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = ChatOpenAI(model=CHAT_MODEL)


def rewrite_question(question, history):
    if len(history) == 0:
        return question

    messages = [("system", REWRITE_PROMPT)]
    for role, text in history:
        messages.append((role, text))
    messages.append(("user", question))
    return llm.invoke(messages).content


def answer_question(question, history):
    search_question = rewrite_question(question, history)
    docs = retriever.invoke(search_question)

    context = ""
    for doc in docs:
        context = context + doc.page_content + "\n\n"

    messages = [("system", SYSTEM_PROMPT)]
    for role, text in history:
        messages.append((role, text))
    messages.append(("user", "Context:\n" + context + "\nQuestion: " + question))

    return llm.invoke(messages).content


if __name__ == "__main__":
    history = []
    q1 = "What does Priya do?"
    a1 = answer_question(q1, history)
    print("Q:", q1)
    print("A:", a1)

    history.append(("user", q1))
    history.append(("assistant", a1))

    q2 = "When did she join?"
    print("Q:", q2)
    print("A:", answer_question(q2, history))