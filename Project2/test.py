import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data
from train import TaxiDriverDataset, evaluate

def test_model(test_dir):
    """
    Initiate the model testing process, including:
    - Loading the saved model
    - Loading and preprocessing test data from test_dir
    - Creating a DataLoader for testing
    - Evaluating the model and printing results
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct the file pattern using test_dir
    test_file_pattern = os.path.join(test_dir, "*.csv")
    # Load test data
    X_test, y_test = load_data(test_file_pattern)

    # Get the device

    test_dataset = TaxiDriverDataset(X_test, y_test, device)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = TaxiDriverClassifier(input_dim=8, output_dim=5).to(device)
    model.load_state_dict(torch.load("./models/best_model.pt"))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accu = evaluate(model, criterion, test_loader, device)
    
    # Print the accuracy in the required format
    print(f"Accuracy={test_accu:.4f}")