def test_lstm_shape():
    from src.models.lstm_model import LSTMPredictor
    import torch
    model = LSTMPredictor(5, 10, 1, 1)
    x = torch.rand((1, 1, 5))
    y = model(x)
    assert y.shape == (1, 1)
