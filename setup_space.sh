#!/usr/bin/env bash
set -euo pipefail

echo "Preparing Hugging Face Space environment..."

mkdir -p data/processed/chroma
mkdir -p data/processed/demo
mkdir -p data/raw/images

echo "Downloading pretrained EfficientNet-B0 weights (this may take a while)..."
python - <<'PY'
import torch
from torchvision import models
model = models.efficientnet_b0(pretrained=True)
torch.save(model.state_dict(), 'data/processed/backbone_b0.pth')
print('Saved data/processed/backbone_b0.pth')
PY

if [ -n "${SAMPLE_IMAGE_URL:-}" ]; then
  echo "Downloading sample image from $SAMPLE_IMAGE_URL"
  curl -L "$SAMPLE_IMAGE_URL" -o data/raw/images/sample.jpg || wget -O data/raw/images/sample.jpg "$SAMPLE_IMAGE_URL"
  echo "Saved sample image to data/raw/images/sample.jpg"
fi

echo "Setup complete. If you need to populate embeddings, run:\n  python -m caretta_reid.database.embedding_store"
