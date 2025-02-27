import unittest
from train import train

class TestTrain(unittest.TestCase):
    def test_train(self):
        train(dataset_csv_path="resource/sample_dataset.csv")


if __name__ == "__main__":
    unittest.main()
