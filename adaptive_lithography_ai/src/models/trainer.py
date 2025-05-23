import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import yaml

from src.models.lstm_model import LSTMPredictor  # Replace with your actual model import

# Load config.yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

checkpoint_dir = config.get("model_checkpoint_dir", "checkpoints")

def train_lstm_model(X_train, y_train, epochs=10, batch_size=32, learning_rate=0.001):
    # Handle pandas data inputs safely
    if hasattr(X_train, "reset_index"):
        X_train = X_train.reset_index(drop=True)
    if hasattr(y_train, "reset_index"):
        y_train = y_train.reset_index(drop=True)

    # Convert to numpy if pandas DataFrame/Series
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    y_train_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train

    # Create dataset and dataloader
    dataset = TensorDataset(
        torch.tensor(X_train_np).float().unsqueeze(1),  # [samples, seq_len=1, features]
        torch.tensor(y_train_np).float().unsqueeze(1)   # [samples, 1]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model params from config
    input_size = config.get("input_size", 4)
    hidden_size = config.get("lstm_hidden_size", 64)
    num_layers = config.get("lstm_num_layers", 2)
    output_size = config.get("lstm_output_size", 1)

    # Initialize model, criterion, optimizer
    model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/lstm_model.pth")
    print(f"Training complete. Model saved at {checkpoint_dir}/lstm_model.pth")
