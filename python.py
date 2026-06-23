import torch
from ultralytics import YOLO

# Check if CUDA (GPU) is available for PyTorch acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch is using device: {device}")

# Load a YOLO26n model (built on PyTorch)
model = YOLO("yolo26n.pt")

# Perform object detection on an image
# PyTorch handles tensor operations and moves data to the GPU automatically
results = model("https://ultralytics.com/images/bus.jpg", device=device)