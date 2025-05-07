import torch
from torchvision import transforms
from PIL import Image

# Class labels
classes = ['Armyworm', 'Cutworm', 'Red Spider Mite']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load("models/inceptionv3.pth", map_location=device)
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_pest(image: Image.Image) -> str:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]
