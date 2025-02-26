import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class TwoTowerModel:
    def __init__(self):
        pass


class Encoder(nn.Module):
    def __init__(self, model_name="line-corporation/line-distilbert-base-japanese"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_str: str):
        output = self.model(**self.tokenizer(input_str, return_tensors="pt"))

        return torch.mean(output.last_hidden_state, dim=1, keepdim=False)
