from langgraph.graph import StateGraph
from langgraph.graph import END

from src.chapter_9.model.state import State
from src.chapter_9.node.answering_node import answering_node
from src.chapter_9.node.check_node import check_node
from src.chapter_9.node.selection_node import selection_node

def build_workflow()->StateGraph:
    workflow = StateGraph(State)

    # ノードの設定追加
    workflow.add_node("selection", selection_node)
    workflow.add_node("answering", answering_node)
    workflow.add_node("check", check_node)

    # 処理の定義
    workflow.set_entry_point("selection")
    workflow.add_edge("selection", "answering")
    workflow.add_edge("answering", "check")

    # 条件付きエッジを定義
    workflow.add_conditional_edges(
        "check",
        lambda state: state.current_judge,
        {True: END, False: "selection"}
    )

    return workflow

# from IPython.display import Image
# Image(compiled.get_graph().draw_png())
