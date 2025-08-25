
from src.chapter_9.model.state import State

from src.chapter_9.build_workflow import build_workflow

def invoke_workflow():

    compiled = build_workflow().compile()

    # 初期状態
    initial_state = State(query="生成AIについて教えてください")
    
    # ワークフローの実行
    result = compiled.invoke(initial_state)

    # 結果
    print(result["messages"][-1])
