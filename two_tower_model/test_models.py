import unittest
from models import TwoTowerModel, Encoder, ContrastiveLoss
import torch


class TestTwoTowerModel(unittest.TestCase):
    def test_constructor(self):
        query_encoder = Encoder()
        doc_encoder = Encoder()

        TwoTowerModel(query_encoder=query_encoder, doc_encoder=doc_encoder)

    def test_forward(self):
        query_encoder = Encoder()
        doc_encoder = Encoder()

        model = TwoTowerModel(query_encoder=query_encoder, doc_encoder=doc_encoder)

        model("query", "doc")


class TestEncoder(unittest.TestCase):
    @unittest.skip("実行に時間がかかるのでスキップ")
    def test_forward(self):
        query_encoder = Encoder()
        output = query_encoder("input")

        self.assertEqual(len(output.shape), 2)


class TestContrastiveLoss(unittest.TestCase):
    def test_forward(self):
        contrastive_loss = ContrastiveLoss()

        y1 = torch.tensor([[0, 0], [0, 0], [0, 0]])
        y2 = torch.tensor([[2, 0], [0, 0], [1, 0]])
        t = torch.tensor([[1, 0, 1]])

        loss = contrastive_loss(y1, y2, t)

        self.assertEqual(loss, 1)


if __name__ == "__main__":
    unittest.main()
