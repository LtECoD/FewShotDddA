python src/embed_protein_seq.py  \
    --pretrained_model Rostlab/prot_t5_xl_uniref50 \
    --seqfile data/correct_seq.tsv \
    --embdir embs/seq \
    --device 0 \
    --batch_size 16