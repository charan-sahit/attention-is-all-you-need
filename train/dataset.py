from pathlib import Path
import sentencepiece as spm
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import pickle
import hashlib

SPM_PATH = Path(__file__).parents[1] / "data" / "bpe.model"
CACHE_DIR = Path(__file__).parents[1] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class WMT14Dataset(Dataset):
    def __init__(self, split, max_len=128, sp_path=SPM_PATH, use_cache=True) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        self.max_len = max_len
        self.bos = self.sp.bos_id()
        self.eos = self.sp.eos_id()

        cache_path = self._cache_path(split, max_len, sp_path)
        if use_cache and cache_path.exists():
            print(f"Loaading from cache path {cache_path}")
            with open(cache_path, "rb") as f:
                self.pairs = pickle.load(f)
            print(f"    {len(self.pairs)} pairs")
            return


        ds = load_dataset("wmt14", "de-en", split=split)
        self.pairs = []
        for i, ex in enumerate(ds):
            src_ids = self.sp.encode(ex["translation"]["en"])

            tgt_ids = self.sp.encode(ex["translation"]["de"])
            if 0 < len(src_ids) <= max_len and 0 < len(tgt_ids) < max_len - 1:
                self.pairs.append((src_ids, tgt_ids))
            if (i+1) % 500_00 == 0:
                print(f"    processed {i + 1:,}")

        if use_cache:
            print(f"Writing cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(self.pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _cache_path(split, max_len, sp_path):
        sp_path = Path(sp_path)
        stat = sp_path.stat()
        key = f"{split}|{max_len}|{sp_path.name}|{int(stat.st_mtime)|{stat.st_size}}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        return CACHE_DIR / f"wmt14_de_en_{split}_{digest}.pkl"
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        return {
            "src" : torch.tensor(src, dtype=torch.long),
            "tgt_in": torch.tensor([self.bos] + tgt, dtype=torch.long),
            "tgt_out": torch.tensor(tgt + [self.eos], dtype=torch.long)
        }


def collate(batch, pad_id=0):
    def pad(seqs):
        m = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    return {
        "src": pad(b["src"] for b in batch),
        "tgt_in": pad(b["tgt_in"] for b in batch),
        "tgt_out": pad(b["tgt_out"] for b in batch)
    }