#!/usr/bin/env python3
# coding: utf-8
import polars as pl

from torch.optim import AdamW
from torch.utils.data import DataLoader

from models import Encoder, TwoTowerModel, ContrastiveLoss
from dataset import PairedTextDataset


def train(dataset_csv_path: str):

    # preparing dataset
    df = pl.read_csv(dataset_csv_path)
    dataset = PairedTextDataset(df=df)

    # preparing dataloader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            shuffle=False,
                            num_workers=2)

    # preparing models, loss and optimizer
    query_encoder = Encoder()
    doc_encoder = Encoder()
    two_tower_model = TwoTowerModel(query_encoder=query_encoder, doc_encoder=doc_encoder)

    contrastive_loss = ContrastiveLoss()

    optimizer = AdamW(two_tower_model.parameters(), lr=0.01, weight_decay=0.01)

    # training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for queries, docs, labels in dataloader:
            h_queries = query_encoder(queries)
            h_docs = doc_encoder(docs)

            loss = contrastive_loss(y1=h_queries, y2=h_docs, t=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def main():
    dataset_csv_path = ""  # FIXME

    train(dataset_csv_path=dataset_csv_path)


if __name__ == "__main__":
    main()
