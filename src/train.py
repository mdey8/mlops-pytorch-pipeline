import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from .model import SimpleCNN
from src.dataset import get_dataloaders

def main():
    with open("configs/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(
        config["dataset"]["data_dir"], config["training"]["batch_size"]
    )

    model = SimpleCNN(
        config["model"]["in_channels"], config["model"]["num_classes"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_loss = float("inf")
    patience_counter = 0
    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    save_path = os.path.join(config["training"]["save_dir"], config["training"]["model_filename"])

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        epoch_loss = train_loss / total
        epoch_acc = correct / total

        # Structured JSON line logging to stdout
        print(json.dumps({"epoch": epoch, "loss": round(epoch_loss, 4), "accuracy": round(epoch_acc, 4)}))

        # Early Stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                print(json.dumps({"event": "early_stopping", "stopped_epoch": epoch}))
                break

if __name__ == "__main__":
    main()