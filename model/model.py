import torch
import torch.nn as nn
from layers import GCN, Transformer, TrajectoryFlowMap

class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, num_pois, embedding_dim, dropout=0.1):
        super(FusionModel, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim, hidden_dim, num_layers, num_pois)  # pass num_pois as num_nodes
        self.transformer = Transformer(hidden_dim, hidden_dim, num_heads, num_layers, dropout)
        self.tfm = TrajectoryFlowMap(num_pois, embedding_dim)
        self.fc = nn.Linear(hidden_dim + embedding_dim, output_dim)

    def forward(self, x):
        # Apply TFM to incorporate global transition patterns
        x_tfm = self.tfm(x)
        
        adj_matrix_learned = self.tfm.transition_matrix  # Use as adj_matrix for GCN
        
        # Apply GCN to capture spatial dependencies
        x_gcn = self.gcn(x, adj_matrix_learned)
        
        # Apply Transformer to capture temporal dependencies
        x_transformer = self.transformer(x_gcn)
      
        # Concatenate the outputs from Transformer and TFM
        x_combined = torch.cat((x_transformer, x_tfm), dim=-1)
        
        # Apply final fully connected layer for output
        output = self.fc(x_combined)
        
        return output