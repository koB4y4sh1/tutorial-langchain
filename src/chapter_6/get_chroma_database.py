
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

from src.chapter_6.get_git_documents import get_git_documents


CLONE_URL="https://github.com/langchain-ai/langchain"
REPO_PATH="./repo/langchain"

def initialize_database(embeddings: OpenAIEmbeddings, persist_directory: str) -> Chroma:
    """データベースを初期化する"""
    os.mkdir(persist_directory)

    # Gitに保存されているドキュメントを取得
    documents = get_git_documents(CLONE_URL, REPO_PATH)

    db = Chroma.from_documents(
        documents, 
        embeddings,
        persist_directory=persist_directory
    )
    return db

def load_database(embeddings: OpenAIEmbeddings, persist_directory: str) -> Chroma:
    """既存のデータベースを読み込む"""
    db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    return db

def get_chroma_database(embeddings: OpenAIEmbeddings, persist_directory: str) -> Chroma:
    """データベースを初期化または既存のものを読み込む"""
    if os.path.exists(persist_directory):
        print("既存のデータベースを読み込み中...")
        db = load_database(embeddings, persist_directory)
        print("既存のデータベースを読み込みました")
        return db
    else:
        print("新しいデータベースを作成中...")
        db = initialize_database(embeddings, persist_directory)
        print("データベースを永続化しました")
        return db