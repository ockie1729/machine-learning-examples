import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class TwoTowerModel(nn.Module):
    def __init__(self, query_encoder: nn.Module, doc_encoder: nn.Module):
        super(TwoTowerModel, self).__init__()

        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def forward(self, query, doc):
        h_queries = self.query_encoder(query)
        h_docs = self.doc_encoder(doc)

        return h_queries, h_docs


class Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "line-corporation/line-distilbert-base-japanese",
        max_length: int = 128,
    ):
        super(Encoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, input_str: str):
        tokenized = self.tokenizer(
            input_str,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
        )
        output = self.model(**tokenized)

        return torch.mean(output.last_hidden_state, dim=1, keepdim=False)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y1, y2, t):
        difference = y1 - y2
        distance_squared = torch.sum(torch.pow(difference, 2), 1)
        distance = torch.sqrt(distance_squared)

        negative_distance = self.margin - distance
        negative_distance = torch.clamp(negative_distance, min=0.0)

        loss = (t * distance_squared + (1 - t) * torch.pow(negative_distance, 2)) / 2.0
        loss = torch.sum(loss) / y1.size()[0]
        return loss
