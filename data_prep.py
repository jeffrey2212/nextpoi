import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_folder = "../dataset/"
output_folder = "dataset/"

def process_gowalla_dataset():
    filename = "Gowalla_totalCheckins.txt"
    datapath = base_folder + filename
    data = pd.read_csv(datapath, delimiter="\t", header=None, 
                    names=["user_id", "timestamp", "latitude", "longitude", "poi_id"])

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    user_mapping = {}
    location_mapping = {}
    user_checkins = {}
    for user_id, poi_id in tqdm(zip(data["user_id"], data["poi_id"]), desc="Building mappings", total=len(data)):
        if user_id not in user_mapping:
            user_mapping[user_id] = len(user_mapping)
            user_checkins[user_id] = 0
        if poi_id not in location_mapping:
            location_mapping[poi_id] = len(location_mapping)
        user_checkins[user_id] += 1

    valid_users = set(user_id for user_id, count in user_checkins.items() if count > 1)
    data = data[data["user_id"].isin(valid_users)]

    user_mapping = {user_id: index for index, user_id in enumerate(valid_users)}

    edge_indices = []
    for _, row in tqdm(data.iterrows(), desc="Creating edge indices", total=len(data)):
        user_id = row["user_id"]
        poi_id = row["poi_id"]
        user_index = user_mapping[user_id]
        poi_index = location_mapping[poi_id]
        edge_indices.append([user_index, poi_index])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)  # Convert to undirected graph

    num_users = len(user_mapping)
    num_locations = len(location_mapping)
    user_checkins_tensor = torch.zeros(num_users, dtype=torch.float)
    for user_id, count in tqdm(user_checkins.items(), desc="Creating node features", total=len(user_checkins)):
        if user_id in valid_users:
            user_index = user_mapping[user_id]
            user_checkins_tensor[user_index] = count

    data = Data(x=user_checkins_tensor, edge_index=edge_index, y=user_checkins_tensor)
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    perm = torch.randperm(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Move data to GPU
    data = data.to(device)

    # Print dataset summary
    print(f"Dataset: Gowalla")
    print("Dataset keys:", data.keys)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of features:", data.num_features)
    print("Feature matrix shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)
    print("Target labels shape:", data.y.shape)

    
    with open(output_folder+'gowalla_graph.pkl', 'wb') as f:
        pickle.dump(data, f)

def process_nyc_dataset():
    filename = "dataset_TSMC2014_NYC.txt"
    datapath = base_folder + filename
    # Read the NYC dataset text file with tab delimiter and no header
    data = pd.read_csv(datapath, delimiter="\t", header=None,
                    names=["user_id", "poi_id", "category_id", "category_name",
                            "latitude", "longitude", "timezone_offset", "timestamp"], encoding='latin1')

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    user_mapping = {}
    location_mapping = {}
    user_checkins = {}
    for user_id, poi_id in tqdm(zip(data["user_id"], data["poi_id"]), desc="Building mappings", total=len(data)):
        if user_id not in user_mapping:
            user_mapping[user_id] = len(user_mapping)
            user_checkins[user_id] = 0
        if poi_id not in location_mapping:
            location_mapping[poi_id] = len(location_mapping)
        user_checkins[user_id] += 1

    valid_users = set(user_id for user_id, count in user_checkins.items() if count > 1)
    data = data[data["user_id"].isin(valid_users)]

    user_mapping = {user_id: index for index, user_id in enumerate(valid_users)}

    edge_indices = []
    for _, row in tqdm(data.iterrows(), desc="Creating edge indices", total=len(data)):
        user_id = row["user_id"]
        poi_id = row["poi_id"]
        user_index = user_mapping[user_id]
        poi_index = location_mapping[poi_id]
        edge_indices.append([user_index, poi_index])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)  # Convert to undirected graph

    num_users = len(user_mapping)
    num_locations = len(location_mapping)
    user_checkins_tensor = torch.zeros(num_users, dtype=torch.float)
    for user_id, count in tqdm(user_checkins.items(), desc="Creating node features", total=len(user_checkins)):
        if user_id in valid_users:
            user_index = user_mapping[user_id]
            user_checkins_tensor[user_index] = count

    data = Data(x=user_checkins_tensor, edge_index=edge_index,  y=user_checkins_tensor.long())

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    perm = torch.randperm(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Move data to GPU
    data = data.to(device)

    # Print dataset summary
    print(f"Dataset: NYC")
    print("Dataset keys:", data.keys)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of features:", data.num_features)
    print("Feature matrix shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)
    print("Target labels shape:", data.y.shape)
   
    with open(output_folder+'nyc_graph.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    #process_gowalla_dataset()
    process_nyc_dataset()