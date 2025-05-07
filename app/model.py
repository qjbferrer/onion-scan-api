import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class labels (replace with your actual class names)
class_labels = ['pest_type_1', 'pest_type_2', 'pest_type_3']  # ← EDIT this

# Function to load InceptionV3 model with pre-trained weights
def load_inceptionv3_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")  # Debugging: Log model loading

    model = models.inception_v3(pretrained=False, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_labels))  # Match number of classes

    state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict):
        try:
            # Load state_dict with strict=False to allow mismatched keys
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise ValueError(f"Error loading state_dict: {e}")
    else:
        raise ValueError("Expected a state_dict, got something else.")
    
    model = model.to(device)
    model.eval()
    return model

# Lazy load the model once
model = None
def get_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "inceptionv3.pth")
        print(f"Model path: {model_path}")  # Debugging: Check model path
        if not os.path.exists(model_path):
            print("Model file not found!")
        model = load_inceptionv3_model(model_path)
    return model

# Predict pest from PIL image
def predict_pest(image: Image.Image) -> str:
    try:
        print("Getting model...")
        model = get_model()
        print("Model loaded.")
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        print("Image transformed.")
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]
        
        print(f"Prediction: {predicted_class}")
        return predicted_class

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "error"
