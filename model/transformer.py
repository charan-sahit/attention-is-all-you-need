import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .layers import Embeddings, PositionalEncoding

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.shared_embed = Embeddings(cfg.vocab_size, cfg.d_model, cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        self.encoder = Encoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.decoder = Decoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.generator = Generator(cfg.d_model, cfg.vocab_size)

        self.generator.proj.weight = self.shared_embed.embed.weight

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.shared_embed(src))
        return self.encoder(x, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.pos_enc(self.shared_embed(tgt))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        out = self.decode(memory, src_mask, tgt, tgt_mask)
        return self.generator(out)