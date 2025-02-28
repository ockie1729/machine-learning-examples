#!/usr/bin/env python3
# coding: utf-8
import polars as pl
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.linalg import vector_norm

from models import Encoder, TwoTowerModel, ContrastiveLoss
from dataset import PairedTextDataset


def calc_accuracy(h_queries, h_docs, labels, margin=1.0) -> float:
    distances = vector_norm(h_queries - h_docs, dim=1)
    predictions = (distances <= margin).int()

    accuracy = torch.eq(predictions, labels).float().mean().item()

    return accuracy


def train(train_dataset_path: str, valid_dataset_path):

    # preparing datasets and dataloaders
    df_train = pl.read_csv(train_dataset_path)
    dataset_train = PairedTextDataset(df=df_train)
    dataloader_train = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=False,
                                 num_workers=2,
                                 drop_last=True)

    df_valid = pl.read_csv(valid_dataset_path)
    dataset_valid = PairedTextDataset(df=df_valid)
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                  batch_size=2,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=True)

    # preparing models, loss and optimizer
    query_encoder = Encoder()
    doc_encoder = Encoder()
    two_tower_model = TwoTowerModel(query_encoder=query_encoder, doc_encoder=doc_encoder)

    contrastive_loss = ContrastiveLoss()

    optimizer = AdamW(two_tower_model.parameters(), lr=0.01, weight_decay=0.01)

    # training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}:")

        two_tower_model.train()
        for queries, docs, labels in tqdm(dataloader_train,
                                          total=len(dataloader_train)):
            h_queries, h_docs = two_tower_model(query=queries, doc=docs)

            loss = contrastive_loss(y1=h_queries, y2=h_docs, t=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}')

        # evaluation
        two_tower_model.eval()
        for queries, docs, labels in tqdm(dataloader_valid,
                                          total=len(dataloader_valid)):
            h_queries, h_docs = two_tower_model(query=queries, doc=docs)

            loss_valid = contrastive_loss(y1=h_queries, y2=h_docs, t=labels)
            acc_valid = calc_accuracy(h_queries=h_queries, h_docs=h_docs, labels=labels)

        print(f'Epoch {epoch+1}/{num_epochs}, Valid Loss: {loss_valid.item()} Valid Acc: {acc_valid}')
        print()

def main():
    dataset_csv_path = ""  # FIXME

    train(dataset_csv_path=dataset_csv_path)


if __name__ == "__main__":
    main()
