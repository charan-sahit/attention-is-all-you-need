from dataclasses import dataclass

@dataclass
class TransformerConfig:
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    n_layers: int = 6
    
    dropout: float = 0.1
    label_smoothing: float = 0.1

    vocab_size: int = 10_000
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # positional encoding
    max_len: int = 5000

    # optimizer
    warmup_steps: int = 4000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9

    # training (IWSLT2017 DE-EN scale)
    total_steps: int = 30_000
    tokens_per_batch: int = 12_000

    # inference
    beam_size: int = 4
    length_penalty: float = 0.6
    max_decode_len: int = 256

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads

    

