import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection
from model import layers


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, src_mask):
        x = self.sublayer1(x, lambda y: self.self_attn(y, y, y, src_mask))
        x = self.sublayer2(x, self.ffn)
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1) -> None:
        super().__init__()        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
