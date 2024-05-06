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
        x = x.long()
        poi_embeddings = self.embedding(x)
        assert not torch.isnan(self.transition_matrix).any() and not torch.isinf(
            self.transition_matrix).any(), "Transition matrix contains NaN or Inf"
        tfm_matrix = torch.softmax(self.transition_matrix, dim=1)
        poi_embeddings_tfm = torch.matmul(tfm_matrix, poi_embeddings)
        return poi_embeddings_tfm
