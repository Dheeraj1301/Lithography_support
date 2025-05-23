import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # Take output of last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

