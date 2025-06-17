import torch
import torch.nn as nn
from torchvision import transforms

NUM_BINS = 101

def load_model():
    print("⚠️ Dummy model loaded")
    
    class DummyAgeModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tensor([25.0])  # always returns 25

    model = DummyAgeModel()
    device = torch.device("cpu")
    return model, device

# Dummy image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(pil_image):
    return transform(pil_image).unsqueeze(0)
