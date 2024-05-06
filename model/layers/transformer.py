import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Change shape to (S, N, E) for MultiheadAttention
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x.permute(1, 0, 2)  # Revert shape to (N, S, E)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(input_dim, output_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    # Define input dimensions and model hyperparameters
    input_dim = 8
    output_dim = 10
    num_heads = 4
    num_layers = 2
    dropout = 0.1

    # Create a sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_dim)

    # Instantiate the Transformer model
    transformer_model = Transformer(input_dim, output_dim, num_heads, num_layers, dropout)

    # Print the model architecture
    print("Transformer Model Architecture:")
    print(transformer_model)

    # Pass the sample input through the Transformer model
    output = transformer_model(x)

    # Print the output shape
    print("\nOutput Shape:")
    print(output.shape)