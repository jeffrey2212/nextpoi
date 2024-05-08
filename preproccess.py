import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

base_folder = "../dataset/"
output_folder = "dataset/"

def process_gowalla_dataset():
    filename = "Gowalla_totalCheckins.txt"
    datapath = base_folder + filename
    
    # Read the Gowalla dataset text file with tab delimiter and no header
    data = pd.read_csv(datapath, delimiter="\t", header=None, 
                    names=["user_id", "timestamp", "latitude", "longitude", "poi_id"])

    # Convert timestamp to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Extract additional temporal features
    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["year"] = data["timestamp"].dt.year

    # Process latitude and longitude
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    # Binning latitude and longitude
    data["lat_bin"] = pd.cut(data["latitude"], bins=20, labels=False)
    data["lon_bin"] = pd.cut(data["longitude"], bins=20, labels=False)

    # Map user_id and poi_id to a continuous range of integers
    unique_users = data['user_id'].unique()
    user_to_index = {user: idx for idx, user in enumerate(unique_users)}
    data['user_id'] = data['user_id'].map(user_to_index)

    unique_pois = data['poi_id'].unique()
    poi_to_index = {poi: idx for idx, poi in enumerate(unique_pois)}
    data['poi_id'] = data['poi_id'].map(poi_to_index)

    # Create a new column "next_poi_id" by shifting the "poi_id" column by -1 within each user's sequence
    data["next_poi_id"] = data.groupby("user_id")["poi_id"].shift(-1)

    # Remove the last check-in for each user since it doesn't have a next POI
    data = data[data["next_poi_id"].notna()]

    # Create a new column "trajectory_id" by combining user_id and check-in sequence
    data["checkin_seq"] = data.groupby("user_id").cumcount() + 1
    data["trajectory_id"] = data["user_id"].astype(str) + "_" + data["checkin_seq"].astype(str)

    # Split the data into features (X) and target variable (y)
    X = data[["user_id", "poi_id", "lat_bin", "lon_bin", "hour", "day", "month", "year", "trajectory_id"]]
    y = data["next_poi_id"].astype(int)  # Ensure next_poi_id is integer

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data to a pickle file
    print("Gowalla: Saving preprocessed data to pickle file")
    with open(output_folder+"gowalla_preprocessed.pkl", "wb") as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
        
    # Save the preprocessed data to CSV files
    print("Gowalla: Saving preprocessed data to CSV files")
    save_to_csv(X_train, y_train, output_folder+"gowalla_train.csv")
    save_to_csv(X_test, y_test, output_folder+"gowalla_test.csv")

def process_nyc_dataset():
    filename = "dataset_TSMC2014_NYC.txt"
    datapath = base_folder + filename

    # Read the NYC dataset text file with tab delimiter and no header
    data = pd.read_csv(datapath, delimiter="\t", header=None,
                    names=["user_id", "poi_id", "category_id", "category_name",
                            "latitude", "longitude", "timezone_offset", "timestamp"], encoding='latin1')

    # Convert timestamp to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%a %b %d %H:%M:%S %z %Y")

    # Extract additional temporal features
    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["year"] = data["timestamp"].dt.year

    # Process latitude and longitude
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    # Binning latitude and longitude
    data["lat_bin"] = pd.cut(data["latitude"], bins=20, labels=False)
    data["lon_bin"] = pd.cut(data["longitude"], bins=20, labels=False)

    # Map POI IDs to a continuous range of integers
    unique_pois = data['poi_id'].unique()
    poi_to_index = {poi: idx for idx, poi in enumerate(unique_pois)}
    data['poi_id'] = data['poi_id'].map(poi_to_index)
    data['next_poi_id'] = data.groupby('user_id')['poi_id'].shift(-1)

    # Remove the last check-in for each user since it doesn't have a next POI
    data = data[data['next_poi_id'].notna()]

    # Create a new column "trajectory_id" by combining user_id and check-in sequence
    data["checkin_seq"] = data.groupby("user_id").cumcount() + 1
    data["trajectory_id"] = data["user_id"].astype(str) + "_" + data["checkin_seq"].astype(str)

    # Split the data into features (X) and target variable (y)
    X = data[["user_id", "poi_id", "lat_bin", "lon_bin", "hour", "day", "month", "year", "trajectory_id"]]
    y = data["next_poi_id"].astype(int)  # Ensure next_poi_id is integer

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data to a pickle file
    print("NYC: Saving preprocessed data to pickle file")
    with open(output_folder + "nyc_preprocessed.pkl", "wb") as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
    
    # Save the preprocessed data to CSV files
    print("NYC: Saving preprocessed data to CSV files")
    save_to_csv(X_train, y_train, output_folder+"nyc_train.csv")
    save_to_csv(X_test, y_test, output_folder+"nyc_test.csv")

def save_to_csv(X, y, filename):
    # Combine the features (X) and target variable (y) into a single DataFrame
    data = pd.concat([X, pd.Series(y, name='next_poi_id')], axis=1)

    # Save the DataFrame to a CSV file
    data.to_csv(filename, index=False)

def main():
    process_gowalla_dataset()
    process_nyc_dataset()

if __name__ == "__main__":
    main()