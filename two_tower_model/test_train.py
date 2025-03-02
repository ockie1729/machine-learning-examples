import unittest
import torch

from train import train, calc_accuracy, test_evaluation, predict
from models import TwoTowerModel, Encoder


class TestTrain(unittest.TestCase):
    @unittest.skip("実行に時間がかかるのでスキップ")
    def test_train(self):
        train(
            train_dataset_path="two_tower_model/resource/sample_dataset.csv",
            valid_dataset_path="two_tower_model/resource/sample_test_dataset.csv",
        )

    def test_calc_accuracy(self):
        prediction = [1, 1, 1, 0, 0]
        label = [1, 1, 0, 1, 1]

        accuracy = calc_accuracy(prediction, label)

        self.assertAlmostEqual(accuracy, 0.4)

    def test_calc_predict(self):
        h_queries = torch.tensor([[0.5, 0.0], [0.1, 0.0], [1.2, 0.0], [-1.2, 0]])
        h_docs = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])
        margin = 1.0

        prediction = predict(h_queries, h_docs, margin=margin)

        self.assertEqual(prediction.numpy().tolist(), [1, 1, 0, 0])

    def test_test_evaluation(self):
        test_dataset_path = "two_tower_model/resource/sample_test_dataset.csv"

        query_encoder = Encoder()
        doc_encoder = Encoder()
        two_tower_model = TwoTowerModel(
            query_encoder=query_encoder, doc_encoder=doc_encoder
        )

        test_evaluation(model=two_tower_model, test_dataset_path=test_dataset_path)


if __name__ == "__main__":
    unittest.main()
