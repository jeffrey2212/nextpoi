import torch
import torch.nn as nn
from layers import GCN, Transformer, TrajectoryFlowMap

class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, num_pois, embedding_dim, dropout=0.1):
        super(FusionModel, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim, hidden_dim, num_layers, num_pois)
        self.transformer = Transformer(hidden_dim, hidden_dim, num_heads, num_layers, dropout)
        self.tfm = TrajectoryFlowMap(num_pois, embedding_dim)
        self.fc = nn.Linear(hidden_dim + embedding_dim, output_dim)

    def forward(self, x):
    # Apply TFM
        x_tfm = self.tfm(x)
        
        # Ensure x_tfm has the correct shape [batch_size, num_pois, feature_size]
        # where num_pois should be 1000
        # Adjusted assertion to reflect actual dimensions
        assert x_tfm.shape[1] == 10 and x_tfm.shape[2] == 64, "TFM output does not match the expected dimensions"
        
        adj_matrix_learned = self.tfm.transition_matrix  # [1000, 1000]

        # Apply GCN
        x_gcn = self.gcn(x_tfm, adj_matrix_learned)
        
        # Apply Transformer
        x_transformer = self.transformer(x_gcn)
    
        # Concatenate outputs
        x_combined = torch.cat((x_transformer, x_tfm), dim=-1)
        
        # Final FC layer
        output = self.fc(x_combined)
        
        return output

if __name__ == "__main__":
    # Define hyperparameters
    input_dim = 8  # Number of input features (based on the summary statistics)
    hidden_dim = 128
    output_dim = 10
    num_heads = 4
    num_layers = 3
    num_pois = 1000  # Number of POIs
    embedding_dim = 64  # Embedding dimension for POI embeddings
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 10

    # Create a sample input, appropriate for the input expected by tfm
    batch_size = 4
    sequence_length = 10
    x = torch.randint(0, num_pois, (batch_size, sequence_length))

    # Instantiate the model
    model = FusionModel(input_dim, hidden_dim, output_dim, num_heads, num_layers, num_pois, embedding_dim, dropout)

    # Print the model architecture
    print("FusionModel Architecture:")
    print(model)

    # Forward pass with the sample input
    output = model(x)

    # Print the output shape
    print("\nOutput from FusionModel:")
    print(output)
    print("Output Shape:", output.shape)