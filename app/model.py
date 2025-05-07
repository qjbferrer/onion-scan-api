import torch
import torch.nn as nn
import torchvision.models as models
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture for InceptionV3
def load_inceptionv3_model(model_path: str):
    # Load the InceptionV3 model without pre-trained weights
    model = models.inception_v3(pretrained=False)
    
    # Check if the model path exists, if not raise an error
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the state_dict (weights)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move model to the device (GPU or CPU)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Example usage
model_path = os.path.join(os.path.dirname(__file__), 'model', 'inceptionv3.pth')  # Relative path
model = load_inceptionv3_model(model_path)
