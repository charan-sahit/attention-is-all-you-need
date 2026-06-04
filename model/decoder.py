import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer1(x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayer2(x, lambda y: self.cross_attn(y, memory, memory, src_mask))
        x = self.sublayer3(x, self.ffn)

        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x