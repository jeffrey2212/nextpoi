import torch
import torch.nn as nn

class TrajectoryFlowMap(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(TrajectoryFlowMap, self).__init__()
        self.embedding = nn.Embedding(num_pois, embedding_dim)
        self.transition_matrix = nn.Parameter(torch.zeros(num_pois, num_pois))

    def forward(self, x):
        poi_embeddings = self.embedding(x)
        tfm_matrix = torch.softmax(self.transition_matrix, dim=1)
        poi_embeddings_tfm = torch.matmul(tfm_matrix, poi_embeddings)
        return poi_embeddings_tfm