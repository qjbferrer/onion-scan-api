import torch
import torch.nn as nn
import torchvision.models as models
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load InceptionV3 model with pre-trained weights from .pth
def load_inceptionv3_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model structure
    model = models.inception_v3(pretrained=False, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # Change to match your class count if not 3

    # Load weights
    state_dict = torch.load(model_path, map_location=device)

    # If the .pth is a pure state_dict
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Expected a state_dict, got something else.")

    model = model.to(device)
    model.eval()
    return model

# Lazy model loading
model = None
def get_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "inceptionv3.pth")
        model = load_inceptionv3_model(model_path)
    return model
