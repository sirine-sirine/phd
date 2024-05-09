import torch
import torch.nn as nn
from multiheadAttention import MultiHeadAttentionBlock
from add_norm import LayerNormalization
from FeedForward import FeedForwardBlock
from EncoderBlock import EncoderBlock
from Encoder import Encoder
from InputEmb import TemporalTransformerInput
from ProjectionLayer import ProjectionLayer
from dataloader import *


class TransEncoder(nn.Module):
    
    def __init__(self, encoder: Encoder, src_embed: TemporalTransformerInput ) -> None:
        """
        

        Parameters
        ----------
        encoder : Encoder
            DESCRIPTION.
        src_embed : TemporalTransformerInput
            DESCRIPTION: 
        projection_layer : ProjectionLayer
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        super().__init__()
        self.encoder = encoder

        self.src_embed = src_embed
        

        #self.projection_layer = projection_layer

    def encode(self, src):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        #src = self.src_pos(src)
        return self.encoder(src)
        
    
    # # def encode(self, src, src_mask):
    # #     src = self.src_embed(src)
    # #     src = self.src_pos + src
    # #     return self.encoder(src, src_mask)
    
    # def project(self, x):
    #      return self.projection_layer(x)
    
    

def build(mask, input_size: int, d_model: int=96, N: int=1, h: int=4, dropout: float=0.1, d_ff: int=2048 ) -> TransEncoder:
    """
    

    Parameters
    ----------
    input_size : int , default = 
        DESCRIPTION. 
    src_seq_len : int
        DESCRIPTION.
    d_model : int, optional, default = 96
        DESCRIPTION. 
    N : int, optional, default = 1 
        DESCRIPTION. 
    h : int, optional, default = 4
        DESCRIPTION. The default is 4.
    dropout : float, optional
        DESCRIPTION. The default is 0.1.
    d_ff : int, optional
        DESCRIPTION. The default is 2048.

    Returns
    -------
    TransEncoder
        DESCRIPTION.

    """
    # Create the embedding layers
    
    print("je suis la ")
    print(d_model, h, dropout)
    input_embedding = TemporalTransformerInput(input_size, d_model)
    
    # Create the encoder blocks
    encoder_blocks = [input_embedding]
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
        
        
        

    encoder = Encoder(mask, d_model, nn.ModuleList(encoder_blocks))

    # Create the input embedding
 

    # Create the projection layer
    #projection_layer = ProjectionLayer(d_model)
    
    # Create the transformer
    transformer = TransEncoder(encoder, input_embedding )
    
    
    return transformer




















