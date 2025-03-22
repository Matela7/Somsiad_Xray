import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
import os

CLASSES = {
    "COVID": 0,
    "NORMAL": 1
}

def load_model():
    model = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    model_path = os.path.join(os.path.dirname(__file__), "model69.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
    for class_name, class_idx in CLASSES.items():
        if class_idx == predicted.item():
            return class_name
    
    return "Unknown"
