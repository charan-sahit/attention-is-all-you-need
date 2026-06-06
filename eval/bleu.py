from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from datasets import load_dataset

from config import TransformerConfig
from infer.translate import load_model, translate_corpus

ROOT = Path(__file__).parents[1]
SPM_PATH = ROOT / "data" / "bpe.model"
CKPT_PATH = ROOT / "checkpoints" / "averaged.pt"

# IWSLT2017 DE→EN: translate German source into English, score against English refs.
HF_DATASET = "iwslt2017"
HF_CONFIG = "iwslt2017-de-en"
SRC_LANG = "de"
TGT_LANG = "en"


def evaluate(ckpt_path=CKPT_PATH, sp_path=SPM_PATH, save_to=None):
    cfg = TransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    model = load_model(ckpt_path, cfg, device)

    test = load_dataset(HF_DATASET, HF_CONFIG, split="test")
    sources = [ex["translation"][SRC_LANG] for ex in test]
    refs = [[ex["translation"][TGT_LANG] for ex in test]]   # list-of-lists for sacrebleu

    hyps = translate_corpus(model, sp, sources, cfg, device)

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print(f"IWSLT2017 DE-EN test BLEU = {bleu.score:.2f}")
    print(f"details: {bleu}")

    if save_to:
        Path(save_to).write_text("\n".join(hyps), encoding="utf-8")
        print(f"    wrote translations to {save_to}")

    return bleu.score


if __name__ == "__main__":
    evaluate(save_to=ROOT / "eval" / "iwslt2017_test.en.hyp")
