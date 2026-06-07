"""
Architecture sanity check. Trains the Transformer on a copy task
(target == source) with a tiny vocab and synthetic data. No external
data needed. Loss should drop from ~ln(50) ≈ 3.9 to <0.5 in ~300 steps.

If this doesn't converge, the model or masking is wrong and there's no
point training on real data.
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import TransformerConfig
from model.transformer import Transformer
from model.masking import make_src_mask, make_tgt_mask


def synthetic_batch(B, T, vocab):
    # Random ids in [3, vocab) so we steer clear of pad/bos/eos.
    src = torch.randint(3, vocab, (B, T))
    src[:, 0] = 1                  # bos at position 0
    tgt = src.clone()              # copy task: target == source
    return src, tgt


def main():
    cfg = TransformerConfig(
        vocab_size=50, d_model=128, d_ff=512,
        n_heads=4, n_layers=2, max_len=64,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(cfg).to(device)
    opt = Adam(model.parameters(), lr=1e-4)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device}  params={n_params:.2f}M")

    final_loss = None
    for step in range(400):
        src, tgt = synthetic_batch(B=16, T=10, vocab=cfg.vocab_size)
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_src_mask(src, cfg.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, cfg.pad_id)

        logp = model(src, tgt_in, src_mask, tgt_mask)
        loss = F.nll_loss(logp.reshape(-1, cfg.vocab_size), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step:>3}  loss {loss.item():.3f}")
        final_loss = loss.item()

    if final_loss is not None and final_loss < 0.5:
        print(f"\n[PASS] copy task converged (loss {final_loss:.3f})")
    else:
        print(f"\n[FAIL] loss stuck at {final_loss:.3f} — inspect masking + weight tying")


if __name__ == "__main__":
    main()
