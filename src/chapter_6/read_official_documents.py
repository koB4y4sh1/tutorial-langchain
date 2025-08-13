
from langchain_openai import OpenAIEmbeddings
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from src.chapter_6.create_hybrid_rag_chain import create_hybrid_rag_chain
from src.chapter_6.create_route_rag_chain import create_route_rag_chain
from src.chapter_6.create_multi_query_rag_chain import create_multi_rag_chain
from src.chapter_6.create_rag_fusion_chain import create_rag_fusion_chain
from src.chapter_6.create_rerank_rag_chain import create_rerank_rag_chain
from src.chapter_6.get_chroma_database import get_chroma_database
from src.chapter_6.create_hyde_rag_chain import create_hyde_rag_chain
from src.chapter_6.create_rag_chain import create_rag_chain

def read_official_documents(
        type=Literal["RAG","HyDE","multi_query","RAG_Fusion","rerank_rag","route_rag","Hybrid"]
    ) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "./db/chapter6"
    
    # データベースの初期化または読み込み
    db:Chroma = get_chroma_database(embeddings, persist_directory)

    prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問: {question}
    ''')

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = db.as_retriever()

    # chain作成（分岐で選択）
    if type == "RAG":
        chain = create_rag_chain(prompt, model, retriever)
    elif type == "HyDE":
        chain = create_hyde_rag_chain(prompt, model, retriever)
    elif type == "multi_query":
        chain = create_multi_rag_chain(prompt, model, retriever)
    elif type == "RAG_Fusion":
        chain = create_rag_fusion_chain(prompt, model, retriever)
    elif type == "rerank_rag":
        chain = create_rerank_rag_chain(prompt, model, retriever)
    elif type == "route_rag":
        chain = create_route_rag_chain(prompt, model, retriever)
    elif type == "Hybrid":
        chain = create_hybrid_rag_chain(prompt, model, retriever)
    else:
        print(f"未知のタイプ: {type}")
        return

    # chain実行
    result = chain.invoke("LangChainの概要を教えて")
    print(result)
    
if __name__ == "__main__":
    read_official_documents("RAG")
