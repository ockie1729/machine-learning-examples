import unittest
from models import TwoTowerModel, Encoder


class TestTwoTowerModel(unittest.TestCase):

    def test_constructor(self):
        model = TwoTowerModel()


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        query_encoder = Encoder()
        output = query_encoder("input")

        self.assertEqual(len(output.shape), 2)


if __name__ == "__main__":
    unittest.main()
