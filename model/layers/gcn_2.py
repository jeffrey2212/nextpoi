import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import pickle
import torch.nn as nn
from torch_sparse import SparseTensor

# Load the preprocessed dataset
with open('dataset/nyc_graph.pkl', 'rb') as f:
    data = pickle.load(f)

# Set float32 matmul precision
torch.set_float32_matmul_precision('medium')

# Define the GCN model for next POI prediction


class GCNNextPOI(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCNNextPOI, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
      #  self.conv2 = GCNConv(hidden_dim, hidden_dim)
       # self.bn2 = BatchNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Convert edge_index to SparseTensor
        edge_index_tensor = SparseTensor(
            row=edge_index[0], col=edge_index[1], sparse_sizes=(x.size(0), x.size(0)))

        x = self.conv1(x, edge_index_tensor)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
     #   x = self.conv2(x, edge_index_tensor)
     #   x = F.relu(x)
     #   x = self.bn2(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        out = self(x, edge_index)
        loss = F.cross_entropy(out[batch.train_mask], y[batch.train_mask])
        self.log('train_loss', loss)

        if (batch_idx + 1) % 4 == 0:
            self.manual_backward(loss)
            self.optimizers().step()
            self.optimizers().zero_grad()
        else:
            self.manual_backward(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        out = self(x, edge_index)
        loss = F.cross_entropy(out[batch.val_mask], y[batch.val_mask])
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        out = self(x, edge_index)
        loss = F.cross_entropy(out[batch.test_mask], y[batch.test_mask])

        pred = out.argmax(dim=1)
        correct = int((pred[batch.test_mask] == y[batch.test_mask]).sum())
        total = int(batch.test_mask.sum())

        if total > 0:
            acc = correct / total
        else:
            acc = 0.0

        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.01, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5)
        return [optimizer], [scheduler]


# Set device to GPU if available and set to float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)  # Convert data to float16
data.x = data.x.to(torch.float16)
# edge_index should remain in long format
data.edge_index = data.edge_index.to(torch.long)
if hasattr(data, 'y'):
    data.y = data.y.to(torch.float16)
# Set model hyperparameters
in_channels = data.num_features
hidden_channels = 16
out_channels = data.num_nodes  # Number of unique POIs

# Create the GCNNextPOI model and convert to float16
model = GCNNextPOI(input_dim=in_channels, hidden_dim=hidden_channels,
                   output_dim=out_channels, dropout=0.5).half()

dataset = [data]
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4,
                        persistent_workers=False, num_workers=0)

# Create a PyTorch Lightning trainer with automatic mixed precision
trainer = pl.Trainer(max_epochs=100, devices="auto", accelerator="auto",
                     precision="16-mixed", accumulate_grad_batches=4)

if __name__ == '__main__':
    # Train the model
    trainer.fit(model, train_dataloaders=dataloader,
                val_dataloaders=dataloader)

    # Test the model
    trainer.test(model, dataloaders=dataloader)
