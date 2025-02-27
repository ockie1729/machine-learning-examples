import unittest
from dataset import PairedTextDataset
import polars as pl

class TestPairedTextDataset(unittest.TestCase):
    def test_constructor(self):
        df = pl.DataFrame({
            "text1": ["aa", "aaa"],
            "text2": ["a", "b"],
            "label": [1, 0],
        })
        
        dataset = PairedTextDataset(df=df)

    def test_len(self):
        df = pl.DataFrame({
            "text1": ["aa", "aaa"],
            "text2": ["a", "b"],
            "label": [1, 0],
        })
        
        dataset = PairedTextDataset(df=df)

        self.assertEqual(len(dataset), 2)

    def test_getitem(self):
        df = pl.DataFrame({
            "text1": ["aa", "aaa"],
            "text2": ["a", "b"],
            "label": [1, 0],
        })

        dataset = PairedTextDataset(df=df)
        text1, text2, label = dataset[1]

        self.assertEqual(text1, "aaa")
        self.assertEqual(text2, "b")
        self.assertEqual(label, 0)


if __name__ == "__main__":
    unittest.main()
