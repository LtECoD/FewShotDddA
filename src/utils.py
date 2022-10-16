import torch

def read_file(fp, skip_header=False, split=False):
    with open(fp, "r") as f:
        lines = f.readlines()
        if skip_header:
            lines = lines[1:]
    if split:
        items = [l.strip().split("\t") for l in lines]
    else:
        items = [l.strip() for l in lines]
    return items


def pad(embs):
    batch_size = len(embs)
    emb_size = embs[0].size(-1)
    max_len = max([emb.size(0) for emb in embs])
    
    padded_embs = torch.zeros(batch_size, max_len, emb_size)
    for idx, emb in enumerate(embs):
        padded_embs[idx, :emb.size(0), :] = emb
    return padded_embs


def write_file(fp, lines, header=None):
    with open(fp, "w") as f:
        if header is not None:
            f.write(header)
        f.writelines(lines)