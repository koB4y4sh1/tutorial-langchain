
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.chapter_6.create_query_generation_chain import create_query_generation


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[str]:
    # 各ドキュメントのコンテンツ (文字列) とそのスコアの対応を保持する辞書を準備
    content_score_mapping = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content

            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            # (1 / (順位 + k)) のスコアを加算
            content_score_mapping[content] += 1 / (rank + k)

    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)  # noqa
    return [content for content, _ in ranked]


def create_rag_fusion_chain(prompt:ChatPromptTemplate, model:ChatOpenAI, retriever: VectorStoreRetriever)->Runnable[any, str]:
    """RAG Fusion"""
    query_generation_chain = create_query_generation(model)

    chain = {
        "question": RunnablePassthrough(),
        "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
    } | prompt | model | StrOutputParser()

    return chain
