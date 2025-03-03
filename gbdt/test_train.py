#!/usr/bin/env python3
# coding: utf-8
import unittest

from train import train


class TrainTest(unittest.TestCase):
    def test_train(self):
        train()
