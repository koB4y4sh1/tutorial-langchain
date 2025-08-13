from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from src.chapter_6.create_query_generation_chain import create_query_generation


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


def create_multi_rag_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """RAGのチェインを作成する"""
    query_generation_chain = create_query_generation(model)

    multi_query_rag_chain = {
        "question": RunnablePassthrough(),
        "context": query_generation_chain | retriever.map(),
    } | prompt | model | StrOutputParser()

    return multi_query_rag_chain