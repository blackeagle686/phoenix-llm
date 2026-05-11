from phoenix_engine import Tensor, nn, optim
from training.scheduler import CosineAnnealingWithWarmup
from core.config import ModelConfig
import time

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        config: ModelConfig,
        lr_max: float = 6e-4,
        lr_min: float = 6e-5,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0
    ):
        self.model = model
        self.config = config
        self.grad_clip = grad_clip
        
        # Initialize Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0, # Will be set by scheduler
            weight_decay=weight_decay
        )
        
        # Initialize Scheduler
        self.scheduler = CosineAnnealingWithWarmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr_max=lr_max,
            lr_min=lr_min
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def train_step(self, input_ids: Tensor, labels: Tensor):
        # 1. Zero Grads
        self.optimizer.zero_grad()
        
        # 2. Forward Pass
        logits = self.model(input_ids) # [B, T, V]
        
        # 3. Compute Loss
        # Flatten for CrossEntropyLoss: [B*T, V] and [B*T]
        B, T, V = logits.shape
        logits_flat = logits.view(B * T, V)
        labels_flat = labels.view(B * T)
        
        loss = self.loss_fn(logits_flat, labels_flat)
        
        # 4. Backward Pass
        loss.backward()
        
        # 5. Gradient Clipping (Note: phoenix-engine might not have global_norm yet)
        # We'll implement a simple per-parameter clip for now if needed
        
        # 6. Optimizer Step
        self.optimizer.step()
        
        # 7. Scheduler Step
        lr = self.scheduler.step()
        
        return loss.item(), lr

    def run(self, steps: int, get_batch_fn):
        self.model.train()
        for step in range(steps):
            start_time = time.time()
            
            input_ids, labels = get_batch_fn()
            loss_val, lr = self.train_step(input_ids, labels)
            
            end_time = time.time()
            ms_per_step = (end_time - start_time) * 1000
            
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss_val:.4f} | LR: {lr:.6f} | {ms_per_step:.2f}ms/step")
