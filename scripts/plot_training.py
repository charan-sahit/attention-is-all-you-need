"""
Parse a training log (the output of `tee logs/train_*.log`) and produce a
3-panel figure: loss, learning rate, throughput — all vs step.

Usage:
    python -m scripts.plot_training [log_path] [out_path]

Defaults to logs/train_interactive.log → logs/training_curves.png.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        # headless safe
import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"step\s+(\d+)\s+loss\s+([\d.]+)\s+lr\s+([\d.eE+\-]+)\s+tok/s/gpu\s+([\d,]+)"
)


def parse(log_path: Path):
    steps, losses, lrs, toks = [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                toks.append(int(m.group(4).replace(",", "")))
    return steps, losses, lrs, toks


def plot(log_path: Path, out_path: Path, warmup_steps: int = 4000):
    steps, losses, lrs, toks = parse(log_path)
    if not steps:
        print(f"No log lines matched in {log_path}")
        return
    print(f"Parsed {len(steps)} log lines (step {steps[0]} → {steps[-1]})")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, losses, lw=1)
    axes[0].set_ylabel("training loss")
    axes[0].set_yscale("log")
    axes[0].axvline(warmup_steps, color="red", linestyle="--", alpha=0.4)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, lrs, lw=1, color="tab:orange")
    axes[1].set_ylabel("learning rate")
    axes[1].axvline(warmup_steps, color="red", linestyle="--", alpha=0.4,
                    label=f"warmup end (step {warmup_steps})")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, toks, lw=1, color="tab:green")
    axes[2].set_ylabel("tok/s/gpu")
    axes[2].set_xlabel("step")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Training curves — {log_path.name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/train_interactive.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("logs/training_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(log_path, out_path)
