import torch
import torch.nn as nn

NUM_BINS = 101  # Sende zaten var

def load_model():
    print("⚠️ Dummy test model loading...")

    class DummyAgeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)  # çok küçük model

        def forward(self, x):
            return torch.tensor([25.0])  # sabit yaş döndürür

    model = DummyAgeModel()
    device = torch.device("cpu")
    return model, device
