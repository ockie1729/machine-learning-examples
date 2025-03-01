from torch.utils.data import Dataset
import polars as pl


class PairedTextDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.text1_list = df["text1"].to_list()
        self.text2_list = df["text2"].to_list()
        self.label_list = df["label"].to_list()

    def __len__(self):
        return len(self.text1_list)

    def __getitem__(self, index: int):
        text1 = self.text1_list[index]
        text2 = self.text2_list[index]
        label = self.label_list[index]

        return text1, text2, label
