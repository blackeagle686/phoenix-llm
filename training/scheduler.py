import math

class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.lr_max * (self.current_step / self.warmup_steps)
        
        if self.current_step > self.total_steps:
            return self.lr_min
            
        # Cosine decay
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine_decay

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        self.optimizer.lr = lr
        return lr
