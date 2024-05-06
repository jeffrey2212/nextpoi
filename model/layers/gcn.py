import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.learned_adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        adj = self.softmax(self.learned_adj)  # Optional: Apply softmax for normalization
        x = torch.matmul(adj, self.fc(x))
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_pois):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # First layer adjusts from the input dimension to the hidden dimension
        self.layers.append(GCNLayer(input_dim, hidden_dim, num_pois))
        # Intermediate layers maintain the hidden dimension
        for _ in range(1, num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, num_pois))
        # Final layer adjusts from the hidden dimension to the output dimension
        self.layers.append(GCNLayer(hidden_dim, output_dim, num_pois))
        
    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x

if __name__ == '__main__':
    # Define input dimensions and model hyperparameters
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    num_layers = 2
    num_nodes = 1000  # Define number of nodes
    num_pois = num_nodes 
    # Create a sample input and adjacency matrix
    batch_size = 4
    x = torch.randn(batch_size, num_nodes, input_dim)
    adj = torch.randn(batch_size, num_nodes, num_nodes)  # Random adjacency matrix

    # Create an instance of the GCN model
    gcn_model = GCN(input_dim, hidden_dim, output_dim, num_layers, num_pois)

    # Print the model architecture
    print("GCN Model Architecture:")
    print(gcn_model)

    # Pass the sample input through the GCN model
    output = gcn_model(x,adj)

    # Print the output shape
    print("\nOutput Shape:")
    print(output.shape)