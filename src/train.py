import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN
from src.dataset import get_dataloaders

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config("configs/training_config.yaml")
    
    # Set seed for reproducibility
    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, _ = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["training"]["batch_size"]
    )

    # Initialize model, loss, optimizer
    model = SimpleCNN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    model.train()
    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save model weights
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, config["training"]["model_filename"])
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train()