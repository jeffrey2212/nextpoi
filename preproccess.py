import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
#import chardet

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

    # Create a new column "next_poi_id" by shifting the "poi_id" column by -1 within each user's sequence
    data["next_poi_id"] = data.groupby("user_id")["poi_id"].shift(-1)

    # Remove the last check-in for each user since it doesn't have a next POI
    data = data[data["next_poi_id"].notna()]

    # Split the data into features (X) and target variable (y)
    X = data[["user_id", "poi_id", "lat_bin", "lon_bin", "hour", "day", "month", "year"]]
    y = data["next_poi_id"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data to a pickle file
    with open(output_folder+"gowalla_preprocessed.pkl", "wb") as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
    
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

    # Create a new column "next_poi_id" by shifting the "poi_id" column by -1 within each user's sequence
    data["next_poi_id"] = data.groupby("user_id")["poi_id"].shift(-1)

    # Remove the last check-in for each user since it doesn't have a next POI
    data = data[data["next_poi_id"].notna()]

    # Split the data into features (X) and target variable (y)
    X = data[["user_id", "poi_id", "lat_bin", "lon_bin", "hour", "day", "month", "year"]]
    y = data["next_poi_id"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data to a pickle file
    with open(output_folder+"nyc_preprocessed.pkl", "wb") as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
    
def main():
    process_gowalla_dataset()
    process_nyc_dataset()

if __name__ == "__main__":
    main()