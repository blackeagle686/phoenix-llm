from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # For GQA, if None, equals num_heads (MHA)
    num_layers: int = 12
    ffn_dim: int = 3072
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_group_size = self.num_heads // self.num_kv_heads
