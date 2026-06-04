import torch

def make_src_mask(src, pad_id=0):
    return (src!=pad_id).unsqueeze(1)

def make_tgt_mask(tgt, pad_id=0):
    B, T = tgt.size()
    pad_mask = (tgt != pad_id).unsqueeze(1)
    causal = torch.tril(torch.ones(T, T, device=tgt.device, dtype=torch.bool))
    causal = causal.unsqueeze(0)
    return pad_mask & causal