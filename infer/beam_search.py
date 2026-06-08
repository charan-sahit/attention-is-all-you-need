import torch
from model.masking import make_src_mask, make_tgt_mask

def length_penalty(length, alpha):
    return ((5+length)/6)*alpha

@torch.no_grad()
def beam_search(model, src_ids, cfg, device):
    model.eval()
    K = cfg.beam_size

    src = src_ids.unsqueeze(0)
    src_mask = make_src_mask(src, cfg.pad_id)
    memory = model.encode(src, src_mask).expand(K, -1, -1)
    src_mask_k = src_mask.expand(K, -1, -1)

    beams = torch.full((K, 1), cfg.bos_id, dtype=torch.long, device=device)
    scores = torch.full((K,),float("-inf"), device=device)
    scores[0] = 0.0

    finished = []
    max_len = min(cfg.max_decode_len, src_ids.size(0)+50)

    for _ in range(max_len):
        tgt_mask = make_tgt_mask(beams, cfg.pad_id)
        out = model.decode(memory, src_mask_k, beams, tgt_mask)
        logp = model.generator(out[:, -1])

        cand = (scores.unsqueeze(1) + logp).view(-1)
        top_scores, top_idx = cand.topk(K)
        beam_idx = top_idx // cfg.vocab_size
        token_idx = top_idx % cfg.vocab_size

        beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores

        for i in (token_idx == cfg.eos_id).nonzero(as_tuple=True)[0].tolist():
            length = beams.size(1) - 1
            lp = length_penalty(length, cfg.length_penalty)
            finished.append((scores[i].item() / lp, beams[i].tolist()))
            scores[i] = float("-inf")
        
        if torch.isinf(scores).all():
            break

    if not finished:
        for i in range(K):
            if not torch.isinf(scores[i]):
                length = beams.size(1) - 1
                lp = length_penalty(length, cfg.length_penalty)
                finished.append((scores[i].item() / lp, beams[i].tolist()))

    finished.sort(key=lambda x: -x[0])
    best = finished[0][1]

    if best and best[0] == cfg.bos_id:
        best = best[1:]
    if cfg.eos_id in best:
        best = best[: best.index(cfg.eos_id)]
    return best

        