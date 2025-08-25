
from langchain_core.prompts import ChatPromptTemplate

from src.chapter_9.model.state import State
from src.chapter_9.model.judgement import Judgement
from src.chapter_9.model.llm_instance import get_llm

def check_node(state: State) -> dict[str, any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
"""以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。
また、その判断理由も説明してください。

ユーザーからの質問: {query}
回答: {answer}
""".strip()
    )
    llm = get_llm()
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }