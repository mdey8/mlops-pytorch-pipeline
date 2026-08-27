import torch
from src.model import SimpleCNN

def test_model_forward():
    model = SimpleCNN(in_channels=1, num_classes=10)
    dummy_input = torch.randn(2, 1, 28, 28)  # Batch of 2 images
    output = model(dummy_input)
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"