import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import TransformerConfig
from model.transformer import Transformer
from model.masking import make_src_mask, make_tgt_mask
from train.dataset import IWSLTDataset, collate
from train.batcher import TokenBucketSampler
from train.schedule import NoamSchedule
from train.loss import LabelSmoothingLoss


def setup_ddp():
    """Initialize torch.distributed. torchrun sets RANK / WORLD_SIZE / LOCAL_RANK.
    Falls back to single-process mode if torchrun env vars are absent."""
    if "RANK" not in os.environ:
        return 0, 0, 1, False
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def cleanup_ddp(active):
    if active:
        dist.destroy_process_group()

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
    rank, local_rank, world_size, ddp_active = setup_ddp()
    is_main = rank == 0

    if ddp_active:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Per-GPU token budget. On 11 GB cards 4K fits comfortably in fp16;
    # effective batch across 3 GPUs ≈ 12K, standard for IWSLT.
    cfg = TransformerConfig(tokens_per_batch=4_000 if ddp_active else 12_000)

    train_ds = IWSLTDataset("train", max_len=128)
    sampler = TokenBucketSampler(
        train_ds,
        tokens_per_batch=cfg.tokens_per_batch,
        num_replicas=world_size,
        rank=rank,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=partial(collate, pad_id=cfg.pad_id),
        num_workers=2,
        pin_memory=True,
    )

    model = Transformer(cfg).to(device)
    n_params = sum(p.numel()  for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Model: {n_params / 1e6:.1f}M parameters, world_size={world_size}")

    if ddp_active:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # The plain underlying module for state_dict / averaging.
    raw_model = model.module if ddp_active else model

    optimizer = Adam(
        model.parameters(),
        lr=0.0,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps
    )
    schedule = NoamSchedule(cfg.d_model, cfg.warmup_steps)
    criterion = LabelSmoothingLoss(cfg.vocab_size, cfg.pad_id, cfg.label_smoothing).to(device)
    # fp16 — 2080 Ti has no bf16 support, so GradScaler is required.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    step = 0
    running_loss = 0
    tokens_seen = 0
    log_every = 100
    ckpt_every = 2500
    keep_last = 5
    epoch = 0

    t0 = time.time()
    model.train()

    while step < cfg.total_steps:
        sampler.set_epoch(epoch)
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
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
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

            if step % log_every == 0 and is_main:
                dt = time.time() - t0
                print(
                    f"step {step:>6}  "
                    f"loss {running_loss / log_every:.3f}  "
                    f"lr {lr:.2e}  "
                    f"tok/s/gpu {tokens_seen / dt:,.0f}"
                )
                running_loss = 0.0

            if step % ckpt_every == 0 and is_main:
                ckpt_path = ckpt_dir / f"ckpt_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                }, ckpt_path)
                ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
                for old in ckpts[:-keep_last]:
                    old.unlink()

            if step >= cfg.total_steps:
                break

        epoch += 1
        if ddp_active:
            dist.barrier()

    if is_main:
        final_ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))[-keep_last:]
        if final_ckpts:
            average_checkpoints(final_ckpts, raw_model)
            torch.save(raw_model.state_dict(), ckpt_dir / "averaged.pt")
            print("Saved averaged.pt")

    cleanup_ddp(ddp_active)

if __name__ == "__main__":
    train()