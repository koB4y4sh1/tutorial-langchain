
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

def create_rag_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """RAGのチェインを作成する"""
    
    chain:Runnable[any, str] = {
        "question": RunnablePassthrough(),
        "context": retriever,
    } | prompt | model | StrOutputParser()

    return chain
