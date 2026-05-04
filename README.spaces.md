# Deploying to Hugging Face Spaces

This repository includes a minimal entrypoint for running on Hugging Face Spaces.

Quick steps:

1. Ensure you have a HF account and an API token (Settings → Access Tokens).
2. Create a new Space on https://huggingface.co/spaces (choose "Gradio" as the SDK).
3. Push this repository (or connect the Space to this GitHub repo). The root `app.py`
   will be used as the start command (`python app.py`).
4. Spaces will install dependencies from `requirements.txt` automatically.

Notes / limitations:
- This repo does not include large dataset files (`data/raw/annotations.json`) — add them
  to the Space storage or load them at runtime from an external URL if needed.
- GPU support on Spaces is optional; depending on your model size you may need a paid plan.
- If your embeddings database must be persistent, configure a remote storage or populate
  the ChromaDB at runtime and persist under `data/processed/chroma`.

If you want, I can prepare a small `setup_space.sh` script to download essential
assets at first-run (e.g., pretrained weights) and wire it into the Space. Let me know.

Quick setup script
------------------

This repo includes `setup_space.sh` which:

- creates the expected `data/processed` and `data/raw/images` folders
- downloads EfficientNet-B0 pretrained weights and stores them at `data/processed/backbone_b0.pth`
- optionally downloads a sample image when you set the `SAMPLE_IMAGE_URL` environment variable

Run on the Space (or locally) after dependencies are installed:

```bash
bash setup_space.sh
```
