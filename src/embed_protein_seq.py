import argparse
import re
import torch
import torch.nn as nn
import numpy as np
import os


from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer


def build_pretrained_model(model_name):
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AlbertModel.from_pretrained(model_name)
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = BertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
        model = XLNetModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unkown model name: {model_name}")
    return tokenizer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--device", type=int, nargs="+")
    parser.add_argument("--seqfile", type=str)
    parser.add_argument("--embdir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    # load model
    print(f">>>>> load pretrained language model {args.pretrained_model}")
    tokenizer, embeder = build_pretrained_model(args.pretrained_model)
    embeder = embeder.eval()
    embeder = embeder.to(args.device[0])
        
    if len(args.device) > 1:
        embeder = embeder.to(args.device[0])
        embeder = nn.DataParallel(module=embeder, device_ids=args.device)

    os.makedirs(args.embdir, exist_ok=True)
    existing_pros = [os.path.splitext(pro_fn)[0] for pro_fn in os.listdir(args.embdir)]

    pro_seq = [l.strip().split("\t") for l in open(args.seqfile, "r").readlines()]
    pro_seq = {p:s for p, s in pro_seq}
    print(f">>>>> Processing {len(pro_seq)} proteins")

    def process_buffer():
        pros = [s[0].strip() for s in buffer]
        seqs = [" ".join(s[1]) for s in buffer]

        inputs = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True)

        inputs = {k: torch.tensor(v).to(args.device[0]) for k, v in inputs.items()}
        with torch.no_grad():
            embedding = embeder(**inputs)
        embedding = embedding.last_hidden_state.cpu().numpy()
        assert len(seqs) == len(pros) == len(embedding)

        for idx in range(len(embedding)):
            seq_len = (inputs['attention_mask'][idx] == 1).sum()
            seq_emb = embedding[idx][:seq_len-1]
            assert seq_len - 1 == len(seqs[idx].strip().split())
            np.save(os.path.join(args.embdir, pros[idx]+".npy"), seq_emb)
            existing_pros.append(pros[idx])

    buffer = []
    processed_num = 0
    for idx, (pro, seq) in enumerate(pro_seq.items()):
        if pro in existing_pros:
            continue
        buffer.append((pro, seq))

        if len(buffer) >= args.batch_size:
            process_buffer()
            processed_num += len(buffer)
            buffer = []

    if len(buffer) > 0:
        process_buffer()
        processed_num += len(buffer)
        buffer = []

    print(f">>>>> Processed {processed_num} proteins. \
        Total {len(os.listdir(args.embdir))} proteins in {args.embdir}.")
    
        
    