
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

def create_hyde_rag_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """Hydeのchainを作成する"""
    hypothetical_prompt = ChatPromptTemplate.from_template("""\
    次の質問に回答する一文を書いてください。

    質問: {question}
    """)

    hypothetical_chain = hypothetical_prompt | model | StrOutputParser()
    hyde_rag_chain = {
        "question": RunnablePassthrough(),
        "context": hypothetical_chain | retriever,
    } | prompt | model | StrOutputParser()
    
    return hyde_rag_chain
