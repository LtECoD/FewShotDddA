import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset

from utils import read_file, write_file, pad


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FSDataset(Dataset):
    def __init__(self, samples, seq_embs, stru_embs, labels=None):
        self.samples = samples
        self.labels = torch.Tensor(labels)
        self.seq_embs = pad([seq_embs[s] for s in samples])
        self.stru_embs = pad([stru_embs[s] for s in samples])
        self.lens = torch.Tensor([seq_embs[s].size(0) for s in samples])

    def get(self):        
        return (self.samples, self.seq_embs, self.stru_embs, self.lens, self.labels)
    
    def __len__(self):
        return len(self.samples)
    

class Reader:
    def __init__(self):
        self.pos_samples = read_file("./data/pos_samples.tsv")
        self.neg_samples = read_file("./data/neg_samples.tsv")

        self.seq_embs = {}
        self.stru_embs = {}
        for n in self.pos_samples + self.neg_samples:
            self.seq_embs[n] = torch.Tensor(np.load(f"./embs/seq/{n}.npy"))
            self.stru_embs[n] = torch.load(f"./embs/stru/{n}_mifst_per_tok.pt")

    def split(self, ratio):
        samples = self.pos_samples + self.neg_samples
        labels = [1] * len(self.pos_samples) + [0] * len(self.neg_samples)

        zipped = list(zip(samples, labels))
        random.shuffle(zipped)        
        samples, labels = list(zip(*zipped))
    
        train_size = int(len(samples) * ratio)
        train_samples = samples[: train_size]
        train_labels = labels[: train_size]
        test_samples = samples[train_size: ]
        test_labels = labels[train_size: ]
        # each split should have at least two positive samples
        assert sum(train_labels) >=2 and sum(test_labels) >= 2

        trainset = FSDataset(
            train_samples,
            seq_embs=self.seq_embs,
            stru_embs=self.stru_embs,
            labels=train_labels)

        testset = FSDataset(
            test_samples,
            seq_embs=self.seq_embs,
            stru_embs=self.stru_embs,
            labels=test_labels)
        return trainset, testset


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        in_dim = int(args.withseq) * args.seqembdim + int(args.withstru) * args.struembdim
        assert in_dim != 0

        #! 对于小样本来说，模型是否过大
        self.module = nn.Sequential(
            nn.Linear(in_dim, args.hiddim),
            nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.hiddim, args.hiddim),
            nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.hiddim, args.hiddim))

    def forward(self, seq_embs, stru_embs, lens):
        """
        Args:
            seq_embs    : bsz x max_len x seq_dim
            stru_embs   : bsz x max_len x stru_dim
        """
        if stru_embs is None:
            embs = seq_embs
        elif seq_embs is None:
            embs = stru_embs
        else:
            embs = torch.cat((seq_embs, stru_embs), dim=-1)
        
        bsz, max_len, _ = embs.size()

        out = self.module(embs)     # bsz x max_len x hid_dim
        hid_dim = out.size(-1)

        mask = torch.arange(max_len).unsqueeze(0).repeat(bsz, 1).to(lens.device)    # bsz x maxlen
        mask = mask >= lens.unsqueeze(-1)                                           # bsz x maxlen
        mask = mask.unsqueeze(-1).repeat(1, 1, hid_dim)                             # bsz x maxlen x hid_dim
        out = torch.masked_fill(out, mask, 0)

        out = torch.sum(out, dim=1)                 # bsz x hid_dim
        out = torch.div(out, lens.unsqueeze(-1))    # bsz x hid_dim
        return out

    def get_score_mat(self, support_out, query_out):
        """
        Args:
            support_out     : snum x hid_dim
            query_out       : qnum x hid_dim
        """

        score_mat = torch.matmul(query_out, support_out.T)      # qnum x snum
        score_mat = torch.exp(score_mat)
        return score_mat

    def get_loss(self, score_mat, labels):
        """
        Args:
            score_mat   : qnum x snum
            labels      : snum
        """
        qnum, snum = score_mat.size()
        assert qnum == snum

        # exclude self scores
        score_mat.fill_diagonal_(0)
        mean_scores = torch.sum(score_mat, dim=-1) / (qnum - 1)    # qnum

        mask = labels.unsqueeze(0).repeat(qnum, 1)
        mask.fill_diagonal_(-1)
        pos_mask = mask == 1
        neg_mask = mask == 0
        
        pos_scores = torch.sum(
            torch.where(pos_mask, score_mat, torch.zeros_like(score_mat)), dim=-1)         # qnum
        mean_pos_scores = pos_scores / torch.sum(pos_mask, dim=-1)  # qnum
        neg_scores = torch.sum(
            torch.where(neg_mask, score_mat, torch.zeros_like(score_mat)), dim=-1)         # qnum
        mean_neg_scores = neg_scores / torch.sum(neg_mask, dim=-1)  # qnum

        # todo
        raise NotImplementedError
    
    def infer(self, score_mat, labels):
        raise NotImplementedError



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument("--withseq", action="store_true", help="use seq embeddings")
    parser.add_argument("--withstru", action="store_true", help="use structure embedding")
    parser.add_argument("--seqembdim", type=int, default=1024)
    parser.add_argument("--struembdim", type=int, default=256)
    parser.add_argument("--hiddim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    # train arguments
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--trainratio", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=int, default=1e-4)

    args = parser.parse_args()
    setup_seed(args.seed)

    trainset, testset = Reader().split(args.trainratio)
    model = Model(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    iterator = tqdm(range(args.steps))
    for idx in iterator:
        samples, seq_embs, stru_embs, lens, labels = trainset.get()
        if not args.withseq:
            seq_embs = None
        if not args.withstru:
            stru_embs = None
        
        # forward model
        out = model(seq_embs, stru_embs, lens)
        score_mat = model.get_score_mat(support_out=out, query_out=out)
        loss = model.get_loss(score_mat=score_mat, labels=labels)
        
        # update parameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validate
        # todo

        
        