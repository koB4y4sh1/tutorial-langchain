from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

# モジュールレベルでシングルトンインスタンスを作成
_llm_instance = ChatOpenAI(model="gpt-4o", temperature=0.0)
# 後からmax_tokensの値を変更できるように、変更可能なフィールドを宣言
_llm = _llm_instance.configurable_fields(max_tokens=ConfigurableField(id='max_tokens'))

def get_llm() -> ChatOpenAI:
    """LLMインスタンスを取得する"""
    return _llm_instance
