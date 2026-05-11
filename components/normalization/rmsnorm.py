from phoenix_engine import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # Fallback to LayerNorm as RMSNorm (no mean subtraction) requires reciprocal/pow ops
        # missing in current phoenix-engine version.
        self.norm = nn.LayerNorm(dim, eps)

    def forward(self, x):
        return self.norm(x)
