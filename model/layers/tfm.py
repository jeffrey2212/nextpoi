import torch
import torch.nn as nn

class TrajectoryFlowMap(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(TrajectoryFlowMap, self).__init__()
        self.embedding = nn.Embedding(num_pois, embedding_dim)
        # Initialize the transition matrix as a learnable parameter with shape [embedding_dim, embedding_dim]
        # This allows transformations within the embedding space.
        self.transition_matrix = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)

    def forward(self, x):
        # Ensure input indices are within the valid range
        assert x.max() < self.embedding.num_embeddings, "Input index out of range for embeddings"
        x = x.long()  # Ensure indices are long integers for embedding lookup
        
        # Get embeddings for input indices
        poi_embeddings = self.embedding(x)

        # Flatten embeddings to [batch_size * sequence_length, embedding_dim] for matrix multiplication
        batch_size, seq_length, embed_dim = poi_embeddings.shape
        poi_embeddings = poi_embeddings.view(-1, embed_dim)

        # Normalize the transition matrix to use as probabilities
        tfm_matrix = torch.softmax(self.transition_matrix, dim=1)
        # Transform embeddings using the learned transition matrix
        poi_embeddings_tfm = torch.matmul(poi_embeddings, tfm_matrix)

        # Reshape transformed embeddings back to the original input shape
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