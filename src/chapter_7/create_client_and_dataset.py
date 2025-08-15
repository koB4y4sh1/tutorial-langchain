

from langsmith import Client
from langsmith.schemas import Dataset

DATASET_NAME = "agent-book"


def create_client_and_dataset() -> tuple[Client, Dataset]:
    """LangSmithのDataSetの作成"""
    # LangSmithのDataSetの作成
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_name=DATASET_NAME)

    return client, client.create_dataset(dataset_name=DATASET_NAME)
