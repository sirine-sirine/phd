import numpy as np 
import torch 
import torch.nn as nn
import math
from dataloader import *




import torch
import torch.nn as nn

class TemporalTransformerInput(nn.Module):
    """
    ceci est un exemple de documentation
    """
    #def __init__(self, input_size, lstm_hidden_size, transformer_input_size):
    def __init__(self, input_size, transformer_input_size):
        """
        

        Parameters
        ----------
        input_size : TYPE
            DESCRIPTION.
        transformer_input_size : TYPE
            DESCRIPTION.
        seq_len : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(TemporalTransformerInput, self).__init__()
        self.input_size = input_size
        #self.lstm_hidden_size = lstm_hidden_size
        self.transformer_input_size = transformer_input_size

        
        # Couche LSTM
        #self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, batch_first=True)
      
        
        # Couche de projection pour ajuster la dimensionnalité à celle du Transformer
        #self.projection = nn.Linear(lstm_hidden_size, transformer_input_size)
        self.projection = nn.Linear(self.input_size, transformer_input_size)
        # Couches de projection pour générer K, Q, V
        self.linear_k = nn.Linear(transformer_input_size, transformer_input_size)
        self.linear_q = nn.Linear(transformer_input_size, transformer_input_size)
        self.linear_v = nn.Linear(transformer_input_size, transformer_input_size)
        
       # self.src_pos = nn.Parameter(torch.zeros(1, self.seq_len ,transformer_input_size))
    def forward(self, x):
        # Passer les séries temporelles à travers la couche LSTM
        #lstm_output, _ = self.lstm(x)
        
        # Projeter la sortie LSTM à la dimension du Transformer
        #transformer_input = self.projection(lstm_output)
        transformer_input = self.projection(x)
        #transformer_input = transformer_input + self.src_pos
        # Générer les matrices K, Q, V
        k = self.linear_k(transformer_input)
        q = self.linear_q(transformer_input)
        v = self.linear_v(transformer_input)
        
        return k, q, v


final_timeseires, final_pearson, labels, site =load_abide_data()
shape= final_timeseires.shape
input_size = shape[-1]
# #lstm_hidden_size = shape[1]
transformer_input_size =96
# seq_len=200

# # Créer l'objet TemporalTransformerInput
temporal_transformer_input = TemporalTransformerInput(input_size, transformer_input_size)

# # Passer les séries temporelles à travers l'objet pour obtenir les matrices K, Q, V
k, q, v = temporal_transformer_input(final_timeseires)

# # Vérifier les dimensions des matrices K, Q, V
print("Matrice K shape:", k.shape)  # Attendu : (1035, 200, 512)
print("Matrice Q shape:", q.shape)  # Attendu : (1035, 200, 512)
print("Matrice V shape:", v.shape)  # Attendu : (1035, 200, 512)


# class TemporalTransformerInput(nn.Module):
#     def __init__(self, input_size, lstm_hidden_size, transformer_input_size):
#         super(TemporalTransformerInput, self).__init__()
#         self.input_size = input_size
#         self.lstm_hidden_size = lstm_hidden_size
#         self.transformer_input_size = transformer_input_size
        
#         # Couche LSTM
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, batch_first=True)
        
#         # Couche de projection pour ajuster la dimensionnalité à celle du Transformer
#         self.projection = nn.Linear(lstm_hidden_size, transformer_input_size)
    
#     def forward(self, x):
#         # Passer les séries temporelles à travers la couche LSTM
#         lstm_output, _ = self.lstm(x)
        
#         # Sélectionner la dernière sortie de la séquence LSTM
#         lstm_output_last = lstm_output[:, -1, :]
        
#         # Projeter la sortie LSTM à la dimension du Transformer
#         transformer_input = self.projection(lstm_output_last)
        
#         return transformer_input



# final_timeseires, final_pearson, labels, site=load_abide_data()
# shape = final_timeseires.shape

# input_size =shape[-1]
# lstm_hidden_size = 10
# transformer_input_size =shape[-1]  #64

# temporal_transformer_input = TemporalTransformerInput(input_size, lstm_hidden_size, transformer_input_size)
# transformer_input = temporal_transformer_input(final_timeseires)
# print(transformer_input.shape)