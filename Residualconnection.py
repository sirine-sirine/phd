
import numpy as np 
import torch 
import torch.nn as nn
import math
from multiheadAttention import MultiHeadAttentionBlock
from add_norm import LayerNormalization
from FeedForward import FeedForwardBlock


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))