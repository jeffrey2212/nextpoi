import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import pickle
import torch.nn as nn
# Load the preprocessed dataset
with open('dataset/nyc_graph.pkl', 'rb') as f:
    data = pickle.load(f)

# Set float32 matmul precision
torch.set_float32_matmul_precision('medium')

# Define the GCN model
class GCN(pl.LightningModule):
   

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        out = self(x, edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        out = self(x, edge_index)
        loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        # Forward pass
        out = self(batch.x, batch.edge_index)
        
        # Print the shapes of the tensors
        print("Output shape:", out.shape)
        print("Test mask shape:", batch.test_mask.shape)
        print("Target shape:", batch.y.shape)
        
        # Compute the loss
        loss = F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])
        
        # Compute the accuracy
        pred = out.argmax(dim=1)
        correct = int((pred[batch.test_mask] == batch.y[batch.test_mask]).sum())
        total = int(batch.test_mask.sum())
        
        if total > 0:
            acc = correct / total
        else:
            acc = 0.0
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# Add an extra dimension to data.x
data.x = data.x.unsqueeze(1)

# Set model hyperparameters
in_channels = data.num_features
hidden_channels = 16
out_channels = 2  # Assuming binary classification (e.g., visited vs. not visited)

# Create the GCN model
model = GCN(input_dim=in_channels, hidden_dim=64, output_dim=2, dropout=0.5)

dataset = [data]
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, persistent_workers=True, num_workers=4)

# Create a PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=300, devices="auto", accelerator="auto")

if __name__ == '__main__':
    # Train the model
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    # Test the model
    trainer.test(model, dataloaders=dataloader)