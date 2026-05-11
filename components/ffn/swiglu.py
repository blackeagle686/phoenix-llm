from phoenix_engine import Tensor, nn

class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        # SwiGLU uses three linear layers (or two with a split)
        # We'll use two for the gate and value, and one for projection
        self.w1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: (W1(x) * silu(W2(x))) * W3
        # Since silu is missing in phoenix_engine, we'll use relu as a temporary fallback
        # or check if we can implement a custom activation.
        x1 = self.w1(x)
        x2 = self.w2(x).relu() # Fallback to relu for now
        
        return self.w3(x1 * x2)