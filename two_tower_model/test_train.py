import unittest
import torch

from train import train, calc_accuracy

class TestTrain(unittest.TestCase):
    @unittest.skip("実行に時間がかかるのでスキップ")
    def test_train(self):
        train(train_dataset_path="resource/sample_dataset.csv",
              valid_dataset_path="resource/sample_test_dataset.csv")


    def test_calc_accuracy(self):
        h_queries = torch.tensor([[0.5, 0.0], [0.1, 0.0], [1.2, 0.0], [-1.2, 0]])
        h_docs = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])
        labels = torch.tensor([1, 0, 0, 0])
        margin = 1.0

        accuracy = calc_accuracy(h_queries, h_docs, labels, margin=margin)

        self.assertEqual(accuracy, 0.75)

if __name__ == "__main__":
    unittest.main()
