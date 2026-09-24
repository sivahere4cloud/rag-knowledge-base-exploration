import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_FILE = PROJECT_ROOT / "practice-kb" / "brightdesk_kb.md"
DB_PATH = PROJECT_ROOT / "vector_db"
EMBED_MODEL = "text-embedding-3-small"


def load_and_chunk():
    text = KB_FILE.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "page"), ("##", "section")],
        strip_headers=False,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    shutil.rmtree(DB_PATH, ignore_errors=True)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH),
    )


def main():
    load_dotenv(override=True)
    chunks = load_and_chunk()
    print("Chunks:", len(chunks))
    vectorstore = build_vectorstore(chunks)
    print("Stored:", vectorstore._collection.count())


if __name__ == "__main__":
    main()