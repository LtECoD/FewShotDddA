from curses.ascii import FS
import os
import random
from setuptools import setup
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import read_file, write_file


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FSDataset(torch.utils.data.Dataset):
    def __init__(self, pos, neg, seq_embs, stru_embs, shuffle=False):
        self.pos = pos
        self.neg = neg
        self.seq_embs = {p: seq_embs[p] for p in pos+neg}
        self.stru_embs = {p: stru_embs[p] for p in pos+neg}

    def get(self):
        pos_seq_embs = [self.seq_embs[s] for s in self.pos]
        pos_stru_embs = [self.stru_embs[s] for s in self.pos]
        neg_seq_embs = [self.seq_embs[s] for s in self.neg]
        neg_stru_embs = [self.stru_embs[s] for s in self.neg]
    
        return (self.pos, pos_seq_embs, pos_stru_embs, self.neg, neg_seq_embs, neg_stru_embs)
    
    def __len__(self):
        return len(self.pos) + len(self.neg)
    

class Reader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.pos_samples = read_file("./data/pos_samples.tsv")
        self.neg_samples = read_file("./data/neg_samples.tsv")

        self.seq_embs = {}
        self.stru_embs = {}

        for n in self.pos_samples + self.neg_samples:
            self.seq_embs[n] = torch.Tensor(np.load(f"./embs/seq/{n}.npy"))
            self.stru_embs[n] = torch.load(f"./embs/stru/{n}_mifst_per_tok.pt")

    def split(self, ratio):
        random.shuffle(self.pos_samples)
        random.shuffle(self.neg_samples)

        train_pos_size = int(len(self.pos_samples) * ratio)
        train_neg_size = int(len(self.neg_samples) * ratio)

        train_pos_samples = self.pos_samples[:train_pos_size]
        train_neg_samples = self.neg_samples[:train_neg_size]
        test_pos_samples = self.pos_samples[train_pos_size: ]
        test_neg_samples = self.neg_samples[train_neg_size: ]

        train_seq_embs = {p: self.seq_embs[p] for p in train_pos_samples+train_neg_samples}
        train_stru_embs = {p: self.stru_embs[p] for p in train_pos_samples+train_neg_samples}
        test_seq_embs = {p: self.seq_embs[p] for p in test_pos_samples+test_neg_samples}
        test_stru_embs = {p: self.stru_embs[p] for p in test_pos_samples+test_neg_samples}
        
        trainset = FSDataset(
            train_pos_samples,
            train_neg_samples,
            train_seq_embs,
            train_stru_embs, 
            bs=self.batch_size,
            shuffle=True)
        testset = FSDataset(
            test_pos_samples,
            test_neg_samples,
            test_seq_embs,
            test_stru_embs,
            bs=-1,
            shuffle=False)
        return trainset, testset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--batchsize", type=int, default=3)
    parser.add_argument("--trainratio", type=float, default=0.7)
    parser.add_argument("--hiddim", type=int)

    args = parser.parse_args()
    setup_seed(args.seed)

    trainset, testset = Reader(args.batchsize).split(args.trainratio)

    for idx in range(10):
        pos, _, _, neg, _, _ = trainset.next()
        print(pos)
        print(neg)
        print()