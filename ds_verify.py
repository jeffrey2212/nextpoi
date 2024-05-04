import pickle

def check_procceed_data(datafile):
    
    print("---Checking "+datafile +"dataset --- \n")
    # Load the preprocessed data from the pickle file
    with open(datafile, "rb") as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    # Print the shapes of the loaded data
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Print the first few rows of X_train and y_train
    print("\nX_train head:")
    print(X_train.head())
    print("\ny_train head:")
    print(y_train.head())

    # Print the data types of the columns in X_train
    print("\nX_train data types:")
    print(X_train.dtypes)

    # Print the unique values and their counts for categorical columns
    print("\nUser ID unique values and counts:")
    print(X_train["user_id"].value_counts())
    print("\nPOI ID unique values and counts:")
    print(X_train["poi_id"].value_counts())
    print("\nLatitude Bin unique values and counts:")
    print(X_train["lat_bin"].value_counts())
    print("\nLongitude Bin unique values and counts:")
    print(X_train["lon_bin"].value_counts())

    # Print the summary statistics of numerical columns
    print("\nX_train summary statistics:")
    print(X_train.describe())
    
    print("---Finish checking "+datafile +"dataset --- \n")

if __name__ == "__main__":
    nyc_datafile = "dataset/nyc_preprocessed.pkl"
    gowalla_datafile = "dataset/gowalla_preprocessed.pkl"
   
    check_procceed_data( nyc_datafile)
    
    check_procceed_data( gowalla_datafile)