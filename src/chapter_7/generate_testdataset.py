
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator, TestDataset
from ragas.testset.evolutions import simple, reasoning, multi_context

def generate_testdataset(documents:list[Document])->TestDataset:
    """合成テストデータ生成の実装."""
    generator = TestsetGenerator.from_langchain(
        generator_llm=ChatOpenAI(model="gpt-4o-mini"),
        critic_llm=ChatOpenAI(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(),
    )

    testset = generator.ｇlangchain_docs(
        documents=documents,
        test_size=4,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
        raise_exceptions=False # Ragas ver0.1.10で発生するExceeptionRunnnerエラー解決。警告メッセージのみ表示させる
    )

    return testset