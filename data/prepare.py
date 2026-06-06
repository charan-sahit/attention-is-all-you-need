import os
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / 'raw'
RAW_DIR.mkdir(exist_ok=True)

SHARED_TXT = RAW_DIR / "wmt14_en_de_shared.txt"
SPM_PREFIX = DATA_DIR / "bpe"
VOCAB_SIZE = 37000

def dump_corpus():
    if SHARED_TXT.exists():
        print(f"Corpus already at {SHARED_TXT}, skipping.")
        return

    ds = load_dataset("wmt_14", "de-en", split="train")
    with open(SHARED_TXT,  "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            en = ex["translation"]["en"].strip().replace("\n", " ")
            de = ex["translation"]["de"].strip().replace("\n", " ")
            if en and de:
                f.write(en + '\n')
                f.write(de + '\n')
            if (i+1) % 500_000 == 0:
                print(f" wrote {i + 1} pairs")
    print(f"Done, corpus at {SHARED_TXT}")

def train_bpe():
    if (SPM_PREFIX.with_suffix(".model")).exists():
        print(f"BPE model already exists at {SPM_PREFIX}.model, skipping training")
        return

    spm.SentencePieceTrainer.train(
        input=str(SHARED_TXT),
        model_prefix=str(SPM_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        pad_piece="<pad>", bos_piece="<s>", eos_piece="</s>", unk_piece="<unk>",
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
    )

    print(f"Trained BPE model at {SPM_PREFIX}.model")


if __name__ == "__main__":
    dump_corpus()
    train_bpe()