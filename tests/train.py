import sys
import os
# Add current directory and phoenix_engine to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "venv", "Lib", "site-packages", "phoenix_engine"))

from phoenix_engine import Tensor
from core.config import ModelConfig
from models.base.model import PhoenixModel
from training.trainer import Trainer
import random

def get_dummy_batch(batch_size: int, seq_len: int, vocab_size: int):
    # Random indices [B, T]
    idx_data = [random.randint(0, vocab_size - 1) for _ in range(batch_size * seq_len)]
    input_ids = Tensor.long(idx_data, shape=[batch_size, seq_len])
    
    # Labels (shifted right, but here just random indices for testing)
    label_data = [random.randint(0, vocab_size - 1) for _ in range(batch_size * seq_len)]
    labels = Tensor.long(label_data, shape=[batch_size, seq_len])
    
    return input_ids, labels

def main():
    print("Initializing Training...")
    config = ModelConfig(
        vocab_size=128,
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        max_seq_len=64
    )
    
    model = PhoenixModel(config)
    
    trainer = Trainer(
        model=model,
        config=config,
        lr_max=1e-3,
        warmup_steps=10,
        total_steps=100
    )
    
    print("Starting Dummy Training Run...")
    
    def batch_fn():
        return get_dummy_batch(batch_size=4, seq_len=32, vocab_size=128)
    
    trainer.run(steps=50, get_batch_fn=batch_fn)
    
    print("Training Verification FINISHED.")

if __name__ == "__main__":
    main()
