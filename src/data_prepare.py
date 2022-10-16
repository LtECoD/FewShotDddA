import os

from utils import read_file, write_file


if __name__ == "__main__":
    """Search duplicate sequences"""
    pro_seqs = dict(read_file("./data/correct_seq.tsv", split=True))
    for p1 in pro_seqs:
        for p2 in pro_seqs:
            if p1 != p2 and pro_seqs[p1] == pro_seqs[p2]:
                print(p1, p2)
    
    pos_pros = [n.split(".")[0] for n in os.listdir("./data/pdb-pos")]
    neg_pros = [n.split(".")[0] for n in os.listdir("./data/pdb-neg")]
    pos_pros = [p for p in pos_pros if p in pro_seqs]
    neg_pros = [p for p in neg_pros if p in pro_seqs]

    lines = [f"{p},{s},{p}.pdb\n" for p, s in pro_seqs.items()]
    write_file("./data/pro_seq_stru.csv", lines, header="name,sequence,pdb\n")

    write_file("./data/pos_samples.tsv", [p+"\n" for p in pos_pros])
    write_file("./data/neg_samples.tsv", [p+"\n" for p in neg_pros])