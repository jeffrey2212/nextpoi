import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, preprocess_data, evaluate_model, print_metrics
from model import FusionModel

def train(model, train_data, train_labels, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        
        # Move data to the device
        train_data = train_data.to(device)
        train_labels = train_labels.to(device)
        
        # Forward pass
        train_output = model(train_data)
        train_loss = criterion(train_output, train_labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}")

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    input_dim = 8  # Number of input features (based on the summary statistics)
    hidden_dim = 128
    output_dim = 10
    num_heads = 4
    num_layers = 2
    num_pois = 1000  # Number of POIs
    embedding_dim = 64  # Embedding dimension for POI embeddings
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 10

    # Load the preprocessed data
    X_train, X_test, y_train, y_test = load_data("nyc_preprocessed.pkl")

    # Preprocess the data
    X_train, y_train, label_encoders = preprocess_data(X_train, y_train)
    X_test, y_test, _ = preprocess_data(X_test, y_test)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create the FusionModel
    model = FusionModel(input_dim, hidden_dim, output_dim, num_heads, num_layers, num_pois, embedding_dim, dropout).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, X_train, y_train, criterion, optimizer, device, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "fusion_model.pth")

    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        test_output = model(X_test)
        test_metrics = evaluate_model(test_output, y_test)
        
    print("Test Metrics:")
    print_metrics(test_metrics)