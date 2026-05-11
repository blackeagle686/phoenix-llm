from phoenix_engine import Tensor, nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(Tensor.randn(max_seq_len, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C]
        T = x.shape[1]
        # Crop to actual sequence length and add to x
        # Use narrow if available in phoenix_engine
        pos = self.pos_emb.narrow(0, 0, T)
        return x + pos