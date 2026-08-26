import io
from PIL import Image
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import yaml

from src.dataset import get_transforms
from src.model import SimpleCNN

# Paths
CONFIG_PATH = Path("configs/training_config.yaml")
DEFAULT_MODEL_PATH = Path(
    r"C:\Users\manos\OneDrive\Documents\GitHub\mlops-pytorch-pipeline\models\model.pt"
)

# Global model state
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Response Schema for Swagger UI
class PredictionResponse(BaseModel):
    filename: str
    probabilities: List[float]
    predicted_class: int


def get_checkpoint_path() -> Path:
    """Resolves checkpoint path from config or fallback path."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        config_path = (
            Path(config["training"]["save_dir"])
            / config["training"]["model_filename"]
        )
        if config_path.exists():
            return config_path

    return DEFAULT_MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for loading model weights on startup."""
    global model
    checkpoint_path = get_checkpoint_path()

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
    else:
        try:
            model = SimpleCNN(in_channels=1, num_classes=10)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )
            model.to(device)
            model.eval()
            print(
                f"Successfully loaded model checkpoint from {checkpoint_path}"
            )
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")

    yield


app = FastAPI(
    title="FashionMNIST Classifier Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Health check endpoint checking if the model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model checkpoint is not loaded"
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predicts class probabilities for an uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0).tolist()
           

        return {
            "filename": file.filename,
            "probabilities": probabilities,
            "predicted_class": int(torch.argmax(outputs, dim=1).item()),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve:app", host="0.0.0.0", port=8080, reload=False)