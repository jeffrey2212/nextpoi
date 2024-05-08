import pandas as pd
import numpy as np
import pickle

# Load the preprocessed data from the CSV file
data = pd.read_csv("dataset/gowalla_train.csv")

# Extract the necessary columns
user_ids = data["user_id"]
poi_ids = data["poi_id"]
trajectory_ids = data["trajectory_id"]

# Create a dictionary to map each unique POI to a unique integer index
unique_pois = poi_ids.unique()
poi_to_index = {poi: index for index, poi in enumerate(unique_pois)}

# Get the number of unique POIs
num_pois = len(unique_pois)

# Initialize an empty adjacency matrix
adjacency_matrix = np.zeros((num_pois, num_pois))

# Iterate over each trajectory and update the adjacency matrix
for _, trajectory_data in data.groupby("trajectory_id"):
    trajectory_pois = trajectory_data["poi_id"]
    for i in range(len(trajectory_pois) - 1):
        current_poi = trajectory_pois.iloc[i]
        next_poi = trajectory_pois.iloc[i + 1]
        current_index = poi_to_index[current_poi]
        next_index = poi_to_index[next_poi]
        adjacency_matrix[current_index, next_index] += 1

# Create a DataFrame to represent the adjacency matrix
adjacency_df = pd.DataFrame(adjacency_matrix, index=unique_pois, columns=unique_pois)

# Save the adjacency matrix as a pickle file
with open("adjacency_matrix.pkl", "wb") as file:
    pickle.dump(adjacency_df, file)

# Save the adjacency matrix as a CSV file
adjacency_df.to_csv("adjacency_matrix.csv")

# Print the adjacency matrix
print(adjacency_df)