import os
import pickle
from sklearn.preprocessing import LabelEncoder


def load_data(data_file):
    dataset_dir = "dataset/"
     
    # Load the preprocessed data from the pickle file
    with open( dataset_dir +data_file, "rb") as file:
        X_train, X_test, y_train, y_test = pickle.load(file)
    return X_train, X_test, y_train, y_test


#def preprocess_data(data, target=None):
    # Create a LabelEncoder for each non-numeric column
#    label_encoders = {}
#    for column in ['user_id', 'poi_id']:
#        label_encoders[column] = LabelEncoder()
#        data[column] = label_encoders[column].fit_transform(data[column])
    
#    if target is not None:
#        label_encoders['target'] = LabelEncoder()
#        target = label_encoders['target'].fit_transform(target)
    
    return data, target, label_encoders