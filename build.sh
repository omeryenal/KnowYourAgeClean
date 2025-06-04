#!/bin/bash

# ✅ Step 1: Install dependencies (especially gdown)
pip install -r requirements.txt

# ✅ Step 2: Download model
echo "Downloading model checkpoint from Google Drive..."
mkdir -p checkpoints
gdown --id 1YLWPywzZX94p8jT5b788VgHenDmItZG2 -O checkpoints/dex_utkface.pt

echo "✅ Model checkpoint downloaded to checkpoints/dex_utkface.pt"
