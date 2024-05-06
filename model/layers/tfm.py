import torch
import torch.nn as nn

class TrajectoryFlowMap(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(TrajectoryFlowMap, self).__init__()
        self.embedding = nn.Embedding(num_pois, embedding_dim)
        # Initialize the transition matrix with dimensions [embedding_dim, embedding_dim]
        self.transition_matrix = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)

    def forward(self, x):
        x = x.long()
        poi_embeddings = self.embedding(x)

        # Flatten embeddings to [batch_size * sequence_length, embedding_dim]
        batch_size, seq_length, embed_dim = poi_embeddings.shape
        poi_embeddings = poi_embeddings.view(-1, embed_dim)

        # Apply softmax to transition matrix and multiply
        tfm_matrix = torch.softmax(self.transition_matrix, dim=1)
        poi_embeddings_tfm = torch.matmul(poi_embeddings, tfm_matrix)  # Dimensionally compatible

        # Reshape back if needed
        poi_embeddings_tfm = poi_embeddings_tfm.view(batch_size, seq_length, -1)

        return poi_embeddings_tfm

if __name__ == '__main__':
    num_pois = 1000
    embedding_dim = 64
    batch_size = 4
    sequence_length = 10
    x = torch.randint(0, num_pois, (batch_size, sequence_length))

    tfm_model = TrajectoryFlowMap(num_pois, embedding_dim)
    print("TrajectoryFlowMap Model Architecture:")
    print(tfm_model)

    output = tfm_model(x)
    print("\nOutput Shape:")
    print(output.shape)