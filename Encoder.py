
import numpy as np 
import torch 
import torch.nn as nn
import math
from add_norm import LayerNormalization


class Encoder(nn.Module):
    def __init__(self, mask, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.mask=mask
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
    





