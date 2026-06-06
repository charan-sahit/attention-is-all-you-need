import time
from functools import partial
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import TransformerConfig
from model.transformer import Transformer
from model.masking import make_src_mask, make_tgt_mask
from train.dataset import WMT14Dataset, collate
from train.batcher import TokenBucketSampler
from train.schedule import NoamSchedule
from train.loss import LabelSmoothingLoss

def average_checkpoints(ckpt_paths, model):
    print(f"Averaging {len(ckpt_paths)} checkpoints...")
    avg=None
    for p in ckpt_paths:
        state =torch.load(p, map_location="cpu")["model"]
        if avg is None:
            avg = {k: v.float().clone() for k,v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].float()
    for k in avg:
        avg[k] /= len(ckpt_paths)
    model.load_state_dict(avg)

def train():
    cfg = TransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = WMT14Dataset("train", max_len=128)
    sampler = TokenBucketSampler(train_ds, tokens_per_batch=cfg.tokens_per_batch)
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=partial(collate, pad_id=cfg.pad_id),
        num_workers=4,
        pin_memory=True,
    )

    model = Transformer(cfg).to(device)
    n_params = sum(p.numel()  for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params / 1e6:.1f}M parameters")

    optimizer = Adam(
        model.parameters(),
        lr=0.0,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps
    )
    schedule = NoamSchedule(cfg.d_model, cfg.warmup_steps)
    criterion = LabelSmoothingLoss(cfg.vocab_size, cfg.pad_id, cfg.label_smoothing).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    step = 0
    running_loss = 0
    tokens_seen = 0
    log_every = 100
    ckpt_every = 2500
    keep_last = 5

    t0 = time.time()
    model.train()

    while step < cfg.total_steps:
        for batch in loader:
            src = batch["src"].to(device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(device, non_blocking=True)
            src_mask = make_src_mask(src, cfg.pad_id)
            tgt_mask = make_tgt_mask(tgt_in, cfg.pad_id)

            step += 1
            lr = schedule.lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logp = model(src, tgt_in, src_mask,tgt_mask)
                loss=criterion(
                    logp.reshape(-1,cfg.vocab_size),
                    tgt_out.reshape(-1)
                )
            scaler.scale(loss).backward()    
            scaler.step(optimizer)
            scaler.update()

            running_loss+= loss.item()
            tokens_seen += (tgt_out != cfg.pad_id).sum().item()

            if step % log_every == 0:
                dt = time.time() - t0
                print(
                    f"step {step:>6} "
                    f"loss {running_loss / log_every:.3f}"
                    f"lr {lr:.2e}"
                    f"tok/s {tokens_seen / dt:,.0f}"
                )
                running_loss = 0.0

            if step % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"ckpt_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                }, ckpt_path)
                ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
                for old in ckpts[:-keep_last]:
                    old.unlink()

            if step >= cfg.total_steps:
                break

    final_ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))[-keep_last:]
    if final_ckpts:
        average_checkpoints(final_ckpts, model)
        torch.save(model.state_dict(), ckpt_dir / "averaged.pt")
        print("Saved average.pt")

if __name__ == "__main__":
    train()