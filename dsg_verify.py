import pickle
from torch.utils.data import Dataset
# Load the preprocessed dataset
with open('dataset/nyc_graph.pkl', 'rb') as f:
    data = pickle.load(f)

print("Dataset keys:", data.keys)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features:", data.num_features)
print("Feature matrix shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)
print("Target labels shape:", data.y.shape)
print("Train mask shape:", data.train_mask.shape)
print("Train mask sum:", data.train_mask.sum())
print("Val mask shape:", data.val_mask.shape)
print("Val mask sum:", data.val_mask.sum())
print("Test mask shape:", data.test_mask.shape)
print("Test mask sum:", data.test_mask.sum())