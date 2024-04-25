"""
We split the video file into fixed length segments
For incomplete segments, we ignore the remaining frames
We use these video segments to extract video embeddings using the imagebind model.
"""
import argparse
import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType

# Load the model from checkpoint into the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def load_and_transform_vision_data(image_paths):
    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


def fetch_image_embeddings(video_path):
    transformed_images = load_and_transform_vision_data(video_path)

    # Initialize a list to hold the outputs
    output_embeddings = []

    # Process the video data in chunks of 100 using tqdm for progress visualization
    with torch.no_grad():
        for i in tqdm(range(0, len(transformed_images), 100), desc="Processing video chunks"):
            chunk = transformed_images[i:i + 100]  # Get the current chunk
            inputs = {
                ModalityType.VISION: chunk
            }
            embeddings = model(inputs)
            output_embeddings.append(embeddings[ModalityType.VISION])

    # Stack all the collected embeddings into a single tensor
    stacked_embeddings = torch.cat(output_embeddings, dim=0)
    print(f"Processed video embeddings for {video_path}")
    return stacked_embeddings


def main():
    video_directory_path = "/data/rohith/captain_cook/videos/resolution_360p/"
    video_feature_directory_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_{int(segment_length)}/"
    os.makedirs(video_feature_directory_path, exist_ok=True)
    video_files = os.listdir(video_directory_path)

    # Use tqdm to show progress bar
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_directory_path, video_file)
        npz_file_path = os.path.join(video_feature_directory_path, video_file + ".npz")

        if os.path.exists(npz_file_path):
            print(f"Skipping {video_file}")
            continue

        # Fetch audio embeddings
        video_embeddings = fetch_image_embeddings(video_path)
        numpy_video_embeddings = video_embeddings.cpu().numpy()
        # Store embeddings in a npz file
        np.savez(npz_file_path, video_embeddings=numpy_video_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_length", type=float, required=True)

    segment_length = parser.parse_args().segment_length
    main()
