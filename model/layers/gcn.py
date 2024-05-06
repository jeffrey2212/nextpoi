import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.learned_adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        adj = self.softmax(self.learned_adj)  # Optional: Apply softmax for normalization
        x = torch.matmul(adj, self.fc(x))
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_nodes):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_dim, hidden_dim, num_nodes))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, num_nodes))
        self.layers.append(GCNLayer(hidden_dim, output_dim, num_nodes))

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x

if __name__ == '__main__':
    # Define input dimensions and model hyperparameters
    input_dim = 8
    hidden_dim = 64
    output_dim = 32
    num_layers = 3

    # Create a sample input and adjacency matrix
    batch_size = 4
    num_nodes = 10
    x = torch.randn(batch_size, num_nodes, input_dim)
    adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

    # Create an instance of the GCN model
    gcn_model = GCN(input_dim, hidden_dim, output_dim, num_layers)

    # Print the model architecture
    print("GCN Model Architecture:")
    print(gcn_model)

    # Pass the sample input through the GCN model
    output = gcn_model(x, adj)

    # Print the output shape
    print("\nOutput Shape:")
    print(output.shape)