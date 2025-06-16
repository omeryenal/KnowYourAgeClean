import os
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from train.train_dex import DEXAgeModel, NUM_BINS

# 📦 Hugging Face repo bilgisi
REPO_ID = "omeryenal/KnowYourAge"
FILENAME = "pytorch_model.bin"

def load_model():
    """
    Loads the pretrained DEXAgeModel from Hugging Face Hub.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model
        device (torch.device): Device used for inference
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    # 🔽 Hugging Face üzerinden indir
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    # Model yükleniyor
    model = DEXAgeModel(num_bins=NUM_BINS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device), device


# Preprocessing function for DEX input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(pil_image):
    """
    Applies preprocessing to the image before feeding it to the model.

    Args:
        pil_image (PIL.Image): Input face image

    Returns:
        torch.Tensor: Preprocessed tensor [1, 3, 224, 224]
    """
    return transform(pil_image).unsqueeze(0)
