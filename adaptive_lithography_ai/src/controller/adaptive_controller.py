import os
import yaml
import torch
from src.models.lstm_model import LSTMPredictor

class AdaptiveLithoController:
    def __init__(self, lstm_model_path="checkpoints/lstm_model.pth"):
        # Load config for architecture
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Use input size from config, must match training input size (4 here)
        input_size = config.get("input_size", 4)
        hidden_size = config.get("lstm_hidden_size", 64)
        num_layers = config.get("lstm_num_layers", 2)
        output_size = config.get("lstm_output_size", 1)

        # Initialize model with correct architecture
        self.model = LSTMPredictor(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, output_size=output_size)

        # Load saved weights
        self.model.load_state_dict(torch.load(lstm_model_path))
        self.model.eval()
        print(f"Loaded LSTM model from {lstm_model_path} successfully!")

    def predict(self, x):
        """
        x: 1D list or array-like input vector of length == input_size
        Returns a single float prediction
        """
        with torch.no_grad():
            x_tensor = torch.tensor(x).float().unsqueeze(0).unsqueeze(1)  # [batch=1, seq_len=1, features=input_size]
            prediction = self.model(x_tensor)
            return prediction.item()

if __name__ == "__main__":
    model_path = "checkpoints/lstm_model.pth"
    controller = AdaptiveLithoController(model_path)

    # Create dummy input matching input_size
    dummy_input = [0.1, 0.2, 0.3, 0.4]  # length must match input_size in config.yaml
    pred = controller.predict(dummy_input)
    print("Prediction on dummy input:", pred)
