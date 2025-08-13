
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def rerank(inp: dict[str,any], top_n:int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return cohere_reranker.compress_documents(documents=documents, query=question)

def create_rerank_rag_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """rerank score by coreha"""
    rerank_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "documents": retriever,
        }
        | RunnablePassthrough.assign(context=rerank)
        | prompt | model | StrOutputParser()
    )
    return rerank_rag_chain
