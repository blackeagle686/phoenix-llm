import sys
import os
# Add current directory to path
sys.path.append(os.getcwd())

from phoenix_engine import Tensor
from core.config import ModelConfig
from models.transformer import LanguageModel

def test_model_init():
    print("Initializing ModelConfig (Small)...")
    config = ModelConfig(
        vocab_size=1000, # Small vocab for testing
        num_layers=2,    # Small layers for testing
        embed_dim=128,
        num_heads=4,
        max_seq_len=64
    )
    
    print("Building LanguageModel...")
    model = LanguageModel(config)
    
    print("Creating dummy input...")
    # [Batch=2, SeqLen=16]
    # In phoenix-engine, we use Tensor.long or random for indices if not using a tokenizer yet
    # Actually, we need integers. Let's use zeros for now if long is not available or just small random.
    idx_data = [0] * (2 * 16)
    idx = Tensor.long(idx_data, shape=[2, 16])
    
    print(f"Forward pass with input shape: {idx.shape}...")
    logits = model(idx)
    
    print(f"Output shape: {logits.shape}")
    expected_shape = [2, 16, config.vocab_size]
    assert list(logits.shape) == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    print("Weight tying check...")
    assert model.lm_head.weight is model.token_embedding.weight, "Weight tying failed!"
    
    print("Architecture Verification SUCCESSFUL!")

if __name__ == "__main__":
    try:
        test_model_init()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
