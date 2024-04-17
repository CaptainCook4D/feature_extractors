from lib.imagebind.imagebind import data
import torch
from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType

image_paths = ["/data/rohith/captain_cook/videos/resolution_360p/10_16_360p.mp4"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.VISION: data.load_and_transform_video_data(image_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)
print(f"Processed video embeddings for {image_paths}")