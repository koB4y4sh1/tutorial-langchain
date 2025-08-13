from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


def create_query_generation(model:ChatOpenAI)->Runnable[any, str]:
    """検索クエリ用のチェインを作成する"""

    query_generation_prompt = ChatPromptTemplate.from_template("""\
    質問に対してベクターデータベースから関連文書を検索するために、
    3つの異なる検索クエリを生成してください。
    距離ベースの類似性検索の限界を克服するために、
    ユーザーの質問に対して複数の視点を提供することが目標です。

    質問: {question}
    """)

    query_generation_chain = (
        query_generation_prompt
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    return query_generation_chain