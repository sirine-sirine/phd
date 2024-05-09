# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:45:20 2024

@author: user
"""
import torch
from InputEmb import TemporalTransformerInput
from dataloader import load_abide_data
from TransEncoder import TransEncoder, build

# def create_padding_mask(seq):
#     # Créer un masque de padding en marquant les positions où les éléments sont nuls
#     mask = (seq != 0).unsqueeze(1)
#     return mask

def main():
    """
    

    Returns
    -------
    None.

    """
    final_timeseires, final_pearson, labels, site =load_abide_data()
    shape= final_timeseires.shape
    input_size = shape[-1]
    
    transformer_input_size =96

    src_seq_len = 200
    d_model = 96
    N = 1
    h = 4
    dropout = 0.1
    d_ff = 2048
    mask = torch.triu(torch.ones((1, src_seq_len, src_seq_len)), diagonal=1).type(torch.int)
    # instanciation de la classe
    temporal_transformer_input = TemporalTransformerInput(input_size, transformer_input_size)
    
    Q, V, K = temporal_transformer_input(final_timeseires)

    
    transformer_model = build(mask, input_size, d_model, N, h, dropout, d_ff)

    
    
    output = transformer_model.encode(Q)
    #output = transformer_model.project(output)
    
    
    
if __name__ == '__main__':
    main()