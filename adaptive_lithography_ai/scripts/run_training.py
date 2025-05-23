from src.data.loader import load_processed_data
from src.models.trainer import train_lstm_model

X_train, y_train = load_processed_data()
train_lstm_model(X_train, y_train)
