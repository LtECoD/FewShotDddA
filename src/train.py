from copy import deepcopy
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


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
    
    def __str__(self):
        pos_samples = [s for idx, s in enumerate(self.samples) if self.labels[idx] == 1]
        neg_samples = [s for idx, s in enumerate(self.samples) if self.labels[idx] == 0]

        return f"{len(pos_samples)} positive samples: {pos_samples}\n{len(neg_samples)} negative samples: {neg_samples}"


class Reader:
    def __init__(self):
        self.pos_samples = read_file("./data/pos_samples.tsv")
        self.neg_samples = read_file("./data/neg_samples.tsv")

        self.seq_embs = {}
        self.stru_embs = {}
        for n in self.pos_samples + self.neg_samples:
            self.seq_embs[n] = torch.Tensor(np.load(f"./embs/seq/{n}.npy"))
            self.stru_embs[n] = torch.load(f"./embs/stru/{n}_mifst_per_tok.pt")

    def split(self, test_pos_num):
        ava_pos = deepcopy(self.pos_samples)                 # sample set is not stable via random seeds
        ava_pos.remove("6U08")
        test_pos = random.sample(ava_pos, test_pos_num)      # 6U08 must in train
        train_pos = list(set(self.pos_samples) - set(test_pos))


        test_neg = random.sample(self.neg_samples, int(len(self.neg_samples)*0.2))      # 4:1 for neg samples
        train_neg = list(set(self.neg_samples) - set(test_neg))
        
        train_samples = train_pos + train_neg
        train_labels = [1] * len(train_pos) + [0] * len(train_neg)
        test_samples = test_pos + test_neg
        test_labels = [1] * len(test_pos) + [0] * len(test_neg)
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
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=args.hiddim, nhead=1, \
                    dim_feedforward=args.hiddim*4, dropout=args.dropout, batch_first=True), 
                num_layers=2)
        )

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

    def get_loss(self, score_mat, support_labels, query_labels, exclude_self=False):
        """
        Args:
            score_mat           : qnum x snum
            support_labels      : snum
            query_labels        : qnum
        """
        qnum, snum = score_mat.size()

        # exclude self scores, fill_diagonal cannot be used as its an in-place operation
        if exclude_self:
            assert qnum == snum
            score_mat = torch.where(
                torch.eye(qnum).bool().to(score_mat.device),
                torch.zeros_like(score_mat),
                score_mat)

        mask = (query_labels.unsqueeze(1) == support_labels.unsqueeze(0)).long().to(score_mat.device) # qnum x snum
        # caculate scores of the same class
        if exclude_self:
            same_mask = torch.where(
                torch.eye(qnum).bool().to(mask.device),
                torch.zeros_like(mask),
                mask)
        else:
            same_mask = mask
        assert (torch.sum(same_mask, dim=-1)>0).all()
        same_scores = torch.sum(
            torch.where(same_mask.bool(), score_mat, torch.zeros_like(score_mat)), dim=-1)  # qnum
        mean_same_scores = same_scores / torch.sum(same_mask, dim=-1)   # qnum
       
        # caculate scores of diff class
        diff_mask = 1 - mask
        if exclude_self:
            assert (torch.diagonal(diff_mask) == 0).all()
        assert (torch.sum(diff_mask, dim=-1)>0).all()
        diff_scores = torch.sum(
            torch.where(diff_mask.bool(), score_mat, torch.zeros_like(score_mat)), dim=-1)
        mean_diff_scores = diff_scores / torch.sum(diff_mask, dim=-1)

        loss = -1. * torch.log(mean_same_scores / (mean_same_scores + mean_diff_scores))
        return torch.mean(loss)
    
    def infer(self, score_mat, support_labels, exclude_self=False):
        qnum, snum = score_mat.size()
        mask = support_labels.unsqueeze(0).repeat(qnum, 1)      # qnum x snum
        if exclude_self:
            mask.fill_diagonal_(-1)

        pos_scores = torch.sum(
            torch.where(mask==1, score_mat, torch.zeros_like(score_mat)), dim=-1)
        pos_mean_scores = pos_scores / torch.sum(mask==1, dim=-1)                   # qnum

        neg_scores = torch.sum(
            torch.where(mask==0, score_mat, torch.zeros_like(score_mat)), dim=-1)
        neg_mean_scores = neg_scores / torch.sum(mask==0, dim=-1)                   # qnum

        pred = (pos_mean_scores > neg_mean_scores).long()
        return pred


def forward_model(args, dataset, model):
    samples, seq_embs, stru_embs, lens, labels = dataset.get()
    if not args.withseq:
        seq_embs = None
    if not args.withstru:
        stru_embs = None
    
    out = model(seq_embs, stru_embs, lens)
    return samples, labels, out


def evaluate(pred, labels):
    pred = pred.numpy()
    labels = labels.numpy()

    precision = precision_score(y_true=labels, y_pred=pred, zero_division=0)
    recall = recall_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    acc = accuracy_score(y_true=labels, y_pred=pred)
    confumat = confusion_matrix(y_true=labels, y_pred=pred)
    return precision, recall, f1, acc, confumat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--testposnum", type=int, default=2)

    # model arguments
    parser.add_argument("--withseq", action="store_true", help="use seq embeddings")
    parser.add_argument("--withstru", action="store_true", help="use structure embedding")
    parser.add_argument("--seqembdim", type=int, default=1024)
    parser.add_argument("--struembdim", type=int, default=256)
    parser.add_argument("--hiddim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    # train arguments
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=int, default=1e-4)

    args = parser.parse_args()
    setup_seed(args.seed)

    trainset, testset = Reader().split(args.testposnum)
    print(f"Trainset:\n{trainset}")
    print()
    print(f"Testset:\n{testset}")
    print()

    model = Model(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    print(model)

    suffix = f"_{args.testposnum}"
    if args.withseq:
        suffix = suffix + "_seq"
    if args.withstru:
        suffix = suffix + "_stru"
    best_loss = 10000.
    for idx in range(args.steps):
        print(f"Epoch: {idx}", end=" ||| ")

        # train
        train_samples, train_labels, train_out = forward_model(args, trainset, model)
        score_mat = model.get_score_mat(support_out=train_out, query_out=train_out)
        loss = model.get_loss(
            score_mat=score_mat, support_labels=train_labels, query_labels=train_labels, exclude_self=True)

        # update parameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Train: loss-{round(float(loss), 4)}", end=" ")

        # evaluate
        pred = model.infer(score_mat, support_labels=train_labels, exclude_self=True)
        precision, recall, f1, acc, confumat = evaluate(pred, train_labels)
        print(f"p-{round(precision, 4)}, r-{round(recall, 4)}, f-{round(f1, 4)}, a-{round(acc, 4)}", end=" ")

        # validate
        model.eval()
        test_samples, test_labels, test_out = forward_model(args, testset, model)
        test_score_mat = model.get_score_mat(support_out=train_out, query_out=test_out)
        test_loss = model.get_loss(
            score_mat=test_score_mat, support_labels=train_labels, query_labels=test_labels)
        print(f"||| Test: loss-{round(float(test_loss), 4)}", end=" ")
        test_pred = model.infer(test_score_mat, support_labels=train_labels)
        precision, recall, f1, acc, confumat = evaluate(test_pred, test_labels)
        print(f"p-{round(precision, 4)}, r-{round(recall, 4)}, f-{round(f1, 4)}, a-{round(acc, 4)}")
        model.train()

        results = [
            f"Step:\t{idx}\n",
            f"Loss:\t{round(float(loss), 4)}\n",
            f"Precision:\t{round(precision, 4)}\n",
            f"Recall:\t{round(recall, 4)}\n",
            f"F1:\t{round(f1, 4)}\n",
            f"Acc:\t{round(acc, 4)}"]
        predictions = [f"{s}\t{int(l)}\t{int(p)}\n" for s, l, p in zip(test_samples, test_labels, test_pred)]

        if float(test_loss) < best_loss:
            torch.save(model.state_dict(), os.path.join("./save", f"best_model{suffix}.ckpt"))
            write_file(os.path.join("./results", f"best_results{suffix}.tsv"), results)
            write_file(os.path.join("./results", f"best_pred{suffix}.tsv"), predictions, header="Protein\tLabel\tPred\n")
            best_loss = float(test_loss)

        if idx == args.steps-1:
            torch.save(model.state_dict(), os.path.join("./save", f"last_model{suffix}.ckpt"))
            write_file(os.path.join("./results", f"last_results{suffix}.tsv"), results)
            write_file(os.path.join("./results", f"last_pred{suffix}.tsv"), predictions, header="Protein\tLabel\tPred\n")