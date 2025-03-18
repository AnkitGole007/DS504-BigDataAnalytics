# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

class TaxiDriverDataset(Dataset):
    """
    Custom dataset class for Taxi Driver Classification.
    Handles loading and preparing data for the model
    """
    def __init__(self, X, y, device):
        
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, optimizer, criterion, train_loader, device):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    model.train()
    total_loss, correct = 0,0

    for batch_X, batch_y in tqdm(train_loader):
      batch_X, batch_y = batch_X.to(device), batch_y.to(device)

      optimizer.zero_grad()
      outputs = model(batch_X)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      correct += (outputs.argmax(dim=1) == batch_y).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)

    return train_loss, train_acc

# Define the testing function
def evaluate(model, criterion, test_loader, device):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    model.eval()
    total_loss, correct = 0,0

    with torch.no_grad():
      for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == batch_y).sum().item()

      test_loss = total_loss / len(test_loader)
      test_acc = correct / len(test_loader.dataset)

    return test_loss, test_acc

def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    X, y = load_data('./data/*.csv')

    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    train_dataset = TaxiDriverDataset(X_train, y_train, device)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = TaxiDriverDataset(X_val,y_val,device)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    model = TaxiDriverClassifier(input_dim=8, output_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(100):
      train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
      val_loss, val_acc = evaluate(model, criterion, val_loader, device)
      
      if (epoch+1)%5 == 0:
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

      if val_acc > best_val_acc:
         best_val_acc = val_acc
         torch.save(model.state_dict(), "./models/best_model.pt")