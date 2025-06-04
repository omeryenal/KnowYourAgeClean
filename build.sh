#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download the model checkpoint from Google Drive
echo "Downloading model checkpoint from Google Drive..."
gdown --id 1YLWPywzZX94p8jT5b788VgHenDmItZG2 -O checkpoints/dex_utkface.pt

echo "✅ Model checkpoint downloaded to checkpoints/dex_utkface.pt"

