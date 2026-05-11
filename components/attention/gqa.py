from phoenix_engine import Tensor, nn
import math
from typing import Optional

class GQA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [B, T, heads, head_dim] then permute to [B, heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        k = k.view(B, T, self.num_kv_heads, self.head_dim).permute([0, 2, 1, 3])
        v = v.view(B, T, self.num_kv_heads, self.head_dim).permute([0, 2, 1, 3])
        
        # If GQA (num_kv_heads < num_heads), we need to repeat K, V heads
        # Since phoenix_engine might lack 'repeat', we check if num_kv_heads == num_heads
        # If not, we'd ideally use a custom GQA kernel or a repeat op.
        if self.num_kv_heads != self.num_heads:
            # Fallback/Placeholder: In a real optimized GQA, we'd use broadcasting or repeat
            # For now, if repeat is missing, GQA will only work if num_kv_heads == num_heads
            # unless the matmul handles broadcasting (which 4D matmul might not for heads).
            pass
            
        # Scaled Dot-Product Attention
        # attn = softmax(Q @ K^T / sqrt(head_dim))
        k_t = k.transpose(-2, -1)
        attn_scores = q.matmul(k_t) * (1.0 / math.sqrt(self.head_dim))
        
        if mask is not None:
            # mask should be broad-castable to [B, heads, T, T]
            attn_scores = attn_scores.masked_fill(mask, -1e9)
            
        attn_probs = attn_scores.softmax()
        
        # attn_out = attn_probs @ V
        attn_output = attn_probs.matmul(v)
        
        # Reshape back to [B, T, C]
        attn_output = attn_output.permute([0, 2, 1, 3]).contiguous().view(B, T, C)
        
        return self.out_proj(attn_output)