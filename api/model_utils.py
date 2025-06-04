import torch
import torch.nn as nn
from torchvision import transforms

# 🧠 Küçük dummy model (sadece test için)
def load_model():
    """
    Returns a minimal PyTorch model and CPU device for testing deployment on Render.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, 1)  # Dummy regression layer
    )
    device = torch.device("cpu")
    return model, device

# 🖼️ Preprocessing function for dummy model (64x64 resize, no normalization)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def preprocess_image(pil_image):
    """
    Preprocesses an image into a [1, 3, 64, 64] tensor.
    
    Args:
        pil_image (PIL.Image): Face image

    Returns:
        torch.Tensor: Preprocessed image
    """
    return transform(pil_image).unsqueeze(0)
