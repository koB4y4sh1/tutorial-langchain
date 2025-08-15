
import nest_asyncio
from langchain_core.documents import Document
from ragas.testset.generator import TestDataset

from src.chapter_7.save_testdataset import save_testdataset
from src.chapter_7.get_git_documents import get_git_documents
from src.chapter_7.generate_testdataset import generate_testdataset


def rag_application():
    documents:list[Document] = get_git_documents()

    # ragasが使用するメタデータである「filename」を設定
    for document in documents:
        document.metadata["filename"] = document.metadata["source"]

    # 合成テストデータ生成の実装
    testset:TestDataset = generate_testdataset(documents)

    testset.to_pandas()

    # LangSmithのDatasetに合成データを保存する
    save_testdataset(testset)