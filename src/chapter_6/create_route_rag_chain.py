from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel

class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


def create_route_rag_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """rerank score by coreha"""
    
    def routed_retriever(inp: dict[str, any]) -> list[Document]:
        question = inp["question"]
        route = inp["route"]
        langchain_document_retriever = retriever.with_config(
            {"run_name": "langchain_document_retriever"}
        )

        web_retriever = TavilySearchAPIRetriever(k=3).with_config(
            {"run_name": "web_retriever"}
        )

        if route == Route.langchain_document:
            return langchain_document_retriever.invoke(question)
        elif route == Route.web:
            return web_retriever.invoke(question)

        raise ValueError(f"Unknown route: {route}")

    route_prompt = ChatPromptTemplate.from_template("""\
        質問に回答するために適切なRetrieverを選択してください。

        質問: {question}
        """)

    route_chain = (
        route_prompt
        | model.with_structured_output(RouteOutput)
        | (lambda x: x.route)
    )
    
    rerank_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "route": route_chain,
        }
        | RunnablePassthrough.assign(context=routed_retriever)
        | prompt | model | StrOutputParser()
    )
    return rerank_rag_chain
