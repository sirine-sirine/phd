import numpy as np 
import torch 
import torch.nn as nn
import math

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None: 
        super().__init__()
        self.proj=nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)