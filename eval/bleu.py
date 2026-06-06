from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from datasets import load_dataset

from config import TransformerConfig
from infer.translate import load_model, translate_corpus
from train.dataset import SPM_PATH

ROOT = Path(__file__).parents[1]
SPM_PATH = ROOT / "data" / "bpe.model"
CKPT_PATH = ROOT / "checkpoints" / "averaged.pt"

def evaluate(ckpt_path=CKPT_PATH, sp_path=SPM_PATH, save_to=None):
    cfg = TransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    model = load_model(ckpt_path, cfg, device)

    test = load_dataset("wmt14", "de-en", split="test")
    sources = [ex["translation"]["en"] for ex in test]
    refs = [ex["translation"]["de"] for ex in test]

    hyps = translate_corpus(model, sp, sources, cfg, device)

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print(f"newstest2014 en-de BLEU = {bleu.score:.2f}")
    print(f"details: {bleu}")

    if save_to:
        Path(save_to).write_text("\n".join(hyps), encoding="utf-8")
        print(f"    wrote translations to {save_to}")
    
    return bleu.score

if __name__ == "__main__":
    evaluate(save_to=ROOT / "eval" / "newstest2014.de.hyp")
