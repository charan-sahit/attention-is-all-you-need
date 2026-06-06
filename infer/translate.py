from pathlib import Path

import sentencepiece as spm
import torch
from tqdm import tqdm

from model.transformer import Transformer
from infer.beam_search import beam_search

def load_model(ckpt_path, cfg, device):
    model = Transformer(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model

def translate_corpus(model, sp, sources, cfg, device):
    out = []
    for text in tqdm(sources, desc="translating"):
        src_ids = sp.encode(text)
        if not src_ids:
            out.append("")
            continue
        src = torch.tensor(src_ids, dtype=torch.long, device=device)
        token_ids = beam_search(model, src, cfg, device)
        out.append(sp.decode(token_ids))
    return out


