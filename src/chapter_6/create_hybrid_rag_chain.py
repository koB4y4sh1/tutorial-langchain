

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from src.chapter_6.create_rag_fusion_chain import reciprocal_rank_fusion
from src.chapter_6.get_git_documents import get_git_documents


CLONE_URL="https://github.com/langchain-ai/langchain"
REPO_PATH="./repo/langchain"

def create_hybrid_rag_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """Hybrid検索用のchainを作成する"""
    documents = get_git_documents(CLONE_URL,REPO_PATH)
    chroma_retriever = retriever.with_config(
        {"run_name": "chroma_retriever"}
    )

    bm25_retriever = BM25Retriever.from_documents(documents).with_config(
        {"run_name": "bm25_retriever"}
    )

    hybrid_retriever = (
        RunnableParallel({
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        })
        | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
        | reciprocal_rank_fusion
    )
    hybrid_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": hybrid_retriever,
        }
        | prompt | model | StrOutputParser()
    )
    
    return hybrid_rag_chain
