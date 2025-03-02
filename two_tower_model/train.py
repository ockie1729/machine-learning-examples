#!/usr/bin/env python3
# coding: utf-8
from typing import List

import argparse

import polars as pl
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.linalg import vector_norm

from models import Encoder, TwoTowerModel, ContrastiveLoss
from dataset import PairedTextDataset


def predict(h_queries, h_docs, margin=1.0):
    distances = vector_norm(h_queries - h_docs, dim=1)
    predictions = (distances <= margin).int()

    return predictions


def calc_accuracy(predictions: List[int], labels: List[int]) -> float:
    accuracy = sum([pred == label for (pred, label) in zip(predictions, labels)]) / len(
        predictions
    )

    return accuracy


def train(
    train_dataset_path: str, valid_dataset_path: str, num_epochs: int = 10
) -> torch.nn.Module:
    # preparing datasets and dataloaders
    df_train = pl.read_csv(train_dataset_path)
    dataset_train = PairedTextDataset(df=df_train)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    df_valid = pl.read_csv(valid_dataset_path)
    dataset_valid = PairedTextDataset(df=df_valid)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    # preparing models, loss and optimizer
    query_encoder = Encoder()
    doc_encoder = Encoder()
    two_tower_model = TwoTowerModel(
        query_encoder=query_encoder, doc_encoder=doc_encoder
    )

    contrastive_loss = ContrastiveLoss()

    optimizer = AdamW(two_tower_model.parameters(), lr=0.01, weight_decay=0.01)

    # training loop
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}:")

        two_tower_model.train()
        for queries, docs, labels in tqdm(
            dataloader_train, total=len(dataloader_train)
        ):
            h_queries, h_docs = two_tower_model(query=queries, doc=docs)

            loss = contrastive_loss(y1=h_queries, y2=h_docs, t=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}")

        # evaluation
        two_tower_model.eval()
        predictions_list = []
        labels_list = []
        for queries, docs, labels in tqdm(
            dataloader_valid, total=len(dataloader_valid)
        ):
            h_queries, h_docs = two_tower_model(query=queries, doc=docs)

            loss_valid = contrastive_loss(y1=h_queries, y2=h_docs, t=labels)
            preds = predict(h_queries=h_queries, h_docs=h_docs)

            predictions_list.append(preds.tolist())
            labels_list.append(labels.tolist())

        predictions_flatten = [pred for sublist in predictions_list for pred in sublist]
        labels_flatten = [label for sublist in labels_list for label in sublist]

        acc_valid = calc_accuracy(
            predictions=predictions_flatten, labels=labels_flatten
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Valid Loss: {loss_valid.item()} Valid Acc: {acc_valid}"
        )
        print()

    return two_tower_model


def test_evaluation(model: torch.nn.Module, test_dataset_path: str) -> float:
    df_test = pl.read_csv(test_dataset_path)
    dataset_test = PairedTextDataset(df=df_test)
    dataloader_test = DataLoader(
        dataset=dataset_test, batch_size=2, shuffle=False, num_workers=2, drop_last=True
    )

    predictions_list = []
    labels_list = []
    for queries, docs, labels in tqdm(dataloader_test, total=len(dataloader_test)):
        h_queries, h_docs = model(query=queries, doc=docs)

        pred = predict(h_queries=h_queries, h_docs=h_docs)
        predictions_list.append(pred.tolist())
        labels_list.append(labels.tolist())

    predictions_flatten = [pred for sublist in predictions_list for pred in sublist]
    labels_flatten = [label for sublist in labels_list for label in sublist]
    acc_test = calc_accuracy(predictions=predictions_flatten, labels=labels_flatten)

    return acc_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset_path", type=str, default="resource/sample_dataset.csv"
    )
    parser.add_argument(
        "--valid_dataset_path", type=str, default="resource/sample_test_dataset.csv"
    )
    parser.add_argument(
        "--test_dataset_path", type=str, default="resource/sample_test_dataset.csv"
    )
    parser.add_argument("--weight_path", type=str, default="output/model.pth")
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()

    two_tower_model = train(
        train_dataset_path=args.train_dataset_path,
        valid_dataset_path=args.valid_dataset_path,
        num_epochs=args.num_epochs,
    )

    accuracy = test_evaluation(
        model=two_tower_model,
        test_dataset_path=args.test_dataset_path,
    )
    print(f"test acc: {accuracy}")

    torch.save(two_tower_model.state_dict(), args.weight_path)


if __name__ == "__main__":
    main()
