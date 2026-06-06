import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_id, smoothing=0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logp, target):
        with torch.nograd():
            true_dist = torch.full_like(logp, self.smoothing / self.vocab_size - 2)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_id] = 0.0
            pad_rows = (target == self.pad_id).unsqueeze(1)
            true_dist.masked_fill_(pad_rows, 0.0)

        loss = -(true_dist * logp).sum(dim=-1)
        n_real = (target != self.pad_id).sum().clamp(min=1)
        return loss.sum() / n_real