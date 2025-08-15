
from ragas.testset.generator import TestDataset

from src.chapter_7.create_client_and_dataset import create_client_and_dataset
def save_testdataset(testset: TestDataset):
    """LangSmithのDatasetに合成データを保存する"""
    
    client, dataset = create_client_and_dataset()

    # 合成テストデータの保存
    inputs = []
    outputs = []
    metadatas = []

    for testset_record in testset.test_data:
        inputs.append(
            {
                "question": testset_record.question,
            }
        )
        outputs.append(
            {
                "contexts": testset_record.contexts,
                "ground_truth": testset_record.ground_truth,
            }
        )
        metadatas.append(
            {
                "source": testset_record.metadata[0]["source"],
                "evolution_type": testset_record.evolution_type,
            }
        )

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadatas,
        dataset_id=dataset.id,
    )