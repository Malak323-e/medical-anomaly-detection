import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

# Transform identique à la validation (pas d'augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

LABELS = ["NORMAL", "ANOMALIE"]

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model_path = os.path.join(os.path.dirname(__file__), "chest_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Modèle entraîné chargé depuis chest_model.pth")
    else:
        # Fallback : poids ImageNet si le modèle n'est pas encore entraîné
        print("ATTENTION : chest_model.pth introuvable, utilisation des poids ImageNet (non entraîné)")
        base = models.resnet18(pretrained=True)
        model.load_state_dict(base.state_dict(), strict=False)

    model.eval()
    return model

model = load_model()

def predict(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred  = torch.argmax(probs).item()
    return {
        "prediction": LABELS[pred],
        "confidence": round(probs[pred].item() * 100, 2),
        "probabilities": {
            "NORMAL":   round(probs[0].item() * 100, 2),
            "ANOMALIE": round(probs[1].item() * 100, 2),
        }
    }
