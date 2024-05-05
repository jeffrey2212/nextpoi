import numpy as np

def evaluate_model(model, test_data, top_k=(5, 10)):
    # Get the user IDs and item IDs from the test data
    user_ids = test_data["user_id"]
    item_ids = test_data["item_id"]
    
    # Get the unique user IDs
    unique_user_ids = np.unique(user_ids)
    
    recalls = {k: [] for k in top_k}
    
    for user_id in unique_user_ids:
        # Get the test items for the current user
        user_test_items = item_ids[user_ids == user_id]
        
        # Generate predictions for the current user
        user_predictions = model.predict(user_id)
        
        # Calculate recall@k for each value of k
        for k in top_k:
            top_k_predictions = user_predictions[:k]
            hits = np.isin(top_k_predictions, user_test_items)
            recall_k = np.sum(hits) / len(user_test_items)
            recalls[k].append(recall_k)
    
    # Calculate the average recall@k for each value of k
    average_recalls = {k: np.mean(recalls[k]) for k in top_k}
    
    return average_recalls

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"Recall@{k}: {v:.4f}")