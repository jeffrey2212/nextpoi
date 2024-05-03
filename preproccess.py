import panda as pd
import pickle 

base_folder = "../dataset/"
output_folder = "dataset/"

def process_gowalla_dataset():
    # Read Gowalla dataset
    gowalla_data = pd.read_csv(base_folder + "Gowalla_totalCheckins.txt")
    
    # Perform data processing operations on Gowalla dataset
    # ...
    
    # Save processed Gowalla dataset as pickle file
    save_as_pickle(gowalla_data, output_folder + "gowalla_processed.pkl")
    
def process_nyc_dataset():
    # Read NYC dataset
    nyc_data = pd.read_csv(base_folder + "dataset_TSMC2014_NYC.txt")
    
    # Perform data processing operations on NYC dataset
    # ...
    
    # Save processed nyc dataset as pickle file
    save_as_pickle(nyc_data, output_folder + "nyc_processed.pkl")
    
def save_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
def main():
    process_gowalla_dataset()
    #process_nyc_dataset()

if __name__ == "__main__":
    main()