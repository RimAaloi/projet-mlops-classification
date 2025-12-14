from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, validator
from typing import List, Optional, Literal
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

app = FastAPI(title="Fashion MNIST Classifier API")

# Environment variables
MODELS_DIR = os.getenv("MODELS_DIR", "models")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "cnn")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Available models mapping
AVAILABLE_MODELS = {
    "mlp": "fashion_classifier.keras",
    "cnn": "cnn_model.keras",
    "transfer": "transfer_model.keras"
}

# Cache for loaded models
loaded_models = {}

# Fashion MNIST class names
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

class PredictionRequest(BaseModel):
    data: List[List[float]]  # List of flattened images (784 pixels)
    model: Optional[Literal["mlp", "cnn", "transfer"]] = None  # Optional model selection

    @validator('data')
    def check_shape(cls, v):
        for img in v:
            if len(img) != 784:
                raise ValueError(f"Each image must have 784 pixels, got {len(img)}")
        return v

class PredictionResponse(BaseModel):
    predictions: List[str]  # Class names
    probabilities: List[List[float]]
    model_used: str

@app.on_event("startup")
def load_default_model():
    """Load the default model on startup."""
    get_model(DEFAULT_MODEL)

def get_model(model_name: str):
    """Load and cache a model by name."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    if model_name not in loaded_models:
        model_path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[model_name])
        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail=f"Model file not found: {model_path}")
        try:
            print(f"Loading model '{model_name}' from {model_path}")
            loaded_models[model_name] = load_model(model_path)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {e}")
    
    return loaded_models[model_name]

@app.get("/health")
def health_check():
    """Check API health and list available models."""
    available = []
    for name, filename in AVAILABLE_MODELS.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            available.append(name)
    
    if not available:
        return {"status": "unhealthy", "reason": "No models found", "models_dir": MODELS_DIR}
    return {"status": "healthy", "available_models": available, "default_model": DEFAULT_MODEL}

@app.get("/models")
def list_models():
    """List all available models and their status."""
    models_info = {}
    for name, filename in AVAILABLE_MODELS.items():
        path = os.path.join(MODELS_DIR, filename)
        models_info[name] = {
            "filename": filename,
            "available": os.path.exists(path),
            "loaded": name in loaded_models
        }
    return {"models": models_info, "default": DEFAULT_MODEL}

@app.get("/model_info")
def model_info(model: str = Query(default=None, description="Model name: mlp, cnn, or transfer")):
    """Get information about a specific model."""
    model_name = model or DEFAULT_MODEL
    m = get_model(model_name)
    
    return {
        "model_name": model_name,
        "model_path": os.path.join(MODELS_DIR, AVAILABLE_MODELS[model_name]),
        "input_shape": m.input_shape,
        "output_shape": m.output_shape
    }

def preprocess_input(data: List[List[float]], target_shape):
    """
    Preprocess input data based on model input shape.
    """
    X = np.array(data)
    
    # 1. Simple MLP: expects (N, 784)
    # 2. CNN: expects (N, 28, 28, 1)
    # 3. Transfer: expects (N, 64, 64, 3)
    
    if target_shape[1:] == (784,):
        # Already flattened
        return X
    
    elif target_shape[1:] == (28, 28, 1):
        # Reshape to 28x28x1
        return X.reshape(-1, 28, 28, 1)
        
    elif target_shape[1:] == (64, 64, 3):
        # Reshape to 28x28x1 first
        X_reshaped = X.reshape(-1, 28, 28, 1)
        # Resize to 64x64
        X_resized = tf.image.resize(X_reshaped, (64, 64))
        # Convert to 3 channels (grayscale to RGB-like)
        X_rgb = tf.concat([X_resized, X_resized, X_resized], axis=-1)
        # Preprocess for MobileNetV2
        return mobilenet_preprocess(X_rgb)
    
    else:
        # Fallback or error
        raise ValueError(f"Unsupported model input shape: {target_shape}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict a single image. Optionally specify which model to use."""
    if len(request.data) != 1:
        raise HTTPException(status_code=400, detail="Endpoint /predict accepts only 1 image. Use /predict_batch for multiple.")

    model_name = request.model or DEFAULT_MODEL
    model = get_model(model_name)
    
    try:
        X = preprocess_input(request.data, model.input_shape)
        probs = model.predict(X)
        pred_indices = np.argmax(probs, axis=1)
        pred_classes = [CLASS_NAMES[i] for i in pred_indices]
        
        return {
            "predictions": pred_classes,
            "probabilities": probs.tolist(),
            "model_used": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=PredictionResponse)
def predict_batch(request: PredictionRequest):
    """Predict multiple images. Optionally specify which model to use."""
    model_name = request.model or DEFAULT_MODEL
    model = get_model(model_name)
    
    try:
        X = preprocess_input(request.data, model.input_shape)
        probs = model.predict(X)
        pred_indices = np.argmax(probs, axis=1)
        pred_classes = [CLASS_NAMES[i] for i in pred_indices]
        
        return {
            "predictions": pred_classes,
            "probabilities": probs.tolist(),
            "model_used": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
