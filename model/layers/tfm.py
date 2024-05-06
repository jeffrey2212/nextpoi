import torch
import torch.nn as nn


class TrajectoryFlowMap(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(TrajectoryFlowMap, self).__init__()
        self.embedding = nn.Embedding(num_pois, embedding_dim)
        # Initialize the transition matrix as a learnable parameter
        self.transition_matrix = nn.Parameter(torch.randn(
            num_pois, num_pois) * 0.01)  # small random initialization

    def forward(self, x):
        x = x.long()  # Ensure input to embedding is of type long
        poi_embeddings = self.embedding(x)
        # Apply softmax to transition matrix to normalize it
        tfm_matrix = torch.softmax(self.transition_matrix, dim=1)
        poi_embeddings_tfm = torch.matmul(tfm_matrix, poi_embeddings)
        return poi_embeddings_tfm
