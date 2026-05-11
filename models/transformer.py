from phoenix_engine import Tensor, nn
from core.config import ModelConfig
from components import GQA, SwiGLU, RMSNorm, PositionalEmbedding
from typing import Optional

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.attn = GQA(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout
        )
        self.ln_2 = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.mlp = SwiGLU(
            embed_dim=config.embed_dim,
            ffn_dim=config.ffn_dim
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Pre-LN architecture
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = PositionalEmbedding(config.max_seq_len, config.embed_dim)
        
        self.blocks = [DecoderBlock(config) for _ in range(config.num_layers)]
        # Register blocks
        for i, block in enumerate(self.blocks):
            self._modules[f"block_{i}"] = block
            
        self.ln_f = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Weight Tying: LM head shares weights with token embeddings
        # phoenix-engine might require explicit assignment
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: Tensor) -> Tensor:
        # idx shape: [B, T]
        B, T = idx.shape
        
        x = self.token_embedding(idx)
        x = self.position_embedding(x)
        
        # Causal mask: [T, T] tril mask
        # phoenix-engine has Tensor.tril
        mask = Tensor.tril(T, T)
        # Invert mask: 0 for allowed, 1 for blocked
        inv_mask = 1.0 - mask
        
        for block in self.blocks:
            x = block(x, mask=inv_mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
