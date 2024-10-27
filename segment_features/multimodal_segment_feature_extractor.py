import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from lib.imagebind.imagebind import ModalityType
from lib.imagebind.imagebind.models import imagebind_model

# Load the model from checkpoint into the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def load_and_transform_rgb_data(image_paths):
    print("Loading and transforming RGB data")
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

    for image_path in tqdm(image_paths):
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


def load_and_transform_depth_data(depth_paths):
    print("Loading and transforming Depth data")
    depth_outputs = []
    for depth_path in tqdm(depth_paths):
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        with open(depth_path, "rb") as fopen:
            image = Image.open(fopen).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        disparity = Image.fromarray(image)
        disparity = data_transform(disparity).to(device)
        depth_outputs.append(disparity)

    return torch.stack(depth_outputs, dim=0)


def extract_rgb_frame_embeddings_for_recording(recording_frames_path):
    transformed_images = load_and_transform_rgb_data(recording_frames_path)

    # Initialize a list to hold the outputs
    output_embeddings = []

    # Process the video data in chunks of 100 using tqdm for progress visualization
    with torch.no_grad():
        for i in tqdm(range(0, len(transformed_images), 30), desc="Processing video chunks"):
            chunk = transformed_images[i:i + 30]  # Get the current chunk
            inputs = {
                ModalityType.VISION: chunk
            }
            embeddings = model(inputs)
            aggregated_embedding = torch.mean(embeddings[ModalityType.VISION], dim=0)
            assert aggregated_embedding.shape[0] == 1024
            output_embeddings.append(aggregated_embedding)

    # Stack all the collected embeddings into a single tensor
    stacked_embeddings = torch.cat(output_embeddings, dim=0)
    print(f"Processed rgb frame embeddings for {recording_frames_path}")
    return stacked_embeddings

def extract_depth_frame_embeddings_for_recording(recording_frames_path):
    transformed_images = load_and_transform_depth_data(recording_frames_path)

    # Initialize a list to hold the outputs
    output_embeddings = []

    # Process the video data in chunks of 100 using tqdm for progress visualization
    with torch.no_grad():
        for i in tqdm(range(0, len(transformed_images), 30), desc="Processing video chunks"):
            chunk = transformed_images[i:i + 30]  # Get the current chunk
            inputs = {
                ModalityType.DEPTH: chunk
            }
            embeddings = model(inputs)
            aggregated_embedding = torch.mean(embeddings[ModalityType.DEPTH], dim=0)
            assert aggregated_embedding.shape[0] == 1024
            output_embeddings.append(aggregated_embedding)

    # Stack all the collected embeddings into a single tensor
    stacked_embeddings = torch.cat(output_embeddings, dim=0)
    print(f"Processed depth frame embeddings for {recording_frames_path}")
    return stacked_embeddings


def extract_depth_frame_embeddings(depth_frames_directory_path, depth_features_path):
    # Get the list of files in the Depth directory
    depth_files = os.listdir(depth_frames_directory_path)

    for depth_file in tqdm(depth_files):
        print("\n ---------------------------------------------------------------")
        print(f"Processing depth file: {depth_file}")

        depth_file_path = os.path.join(depth_frames_directory_path, depth_file)
        depth_frames_paths_list = [os.path.join(depth_file_path, f) for f in os.listdir(depth_file_path)]
        depth_frame_embeddings = extract_depth_frame_embeddings_for_recording(depth_frames_paths_list[:5])
        depth_features_file_path = os.path.join(depth_features_path, f"{depth_file}.npz")
        np.savez(depth_features_file_path, embeddings=depth_frame_embeddings.cpu().numpy())


def extract_rgb_frame_embeddings(rgb_frames_directory_path, rgb_features_path):
    # Get the list of files in the RGB directory
    rgb_files = os.listdir(rgb_frames_directory_path)

    for rgb_file in tqdm(rgb_files):
        print("\n ---------------------------------------------------------------")
        print(f"Processing rgb file: {rgb_file}")

        rgb_file_path = os.path.join(rgb_frames_directory_path, rgb_file)
        rgb_frames_path_list = [os.path.join(rgb_file_path, f) for f in os.listdir(rgb_file_path)]
        rgb_frame_embeddings = extract_rgb_frame_embeddings_for_recording(rgb_frames_path_list)
        rgb_features_file_path = os.path.join(rgb_features_path, f"{rgb_file}.npz")
        np.savez(rgb_features_file_path, embeddings=rgb_frame_embeddings.cpu().numpy())



def main():
    # Paths corresponding to each modality
    depth_frames_directory_path = "/data/rohith/captain_cook/depth/gopro/resolution_360p"
    rgb_frames_directory_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p"

    # Path to store the multimodal embeddings
    imagebind_features_directory_path = "/data/rohith/captain_cook/features/gopro/segments/imagebind"
    os.makedirs(imagebind_features_directory_path, exist_ok=True)

    rgb_features_path = os.path.join(imagebind_features_directory_path, "rgb")
    os.makedirs(rgb_features_path, exist_ok=True)

    depth_features_path = os.path.join(imagebind_features_directory_path, "depth")
    os.makedirs(depth_features_path, exist_ok=True)

    # extract_rgb_frame_embeddings(rgb_frames_directory_path, rgb_features_path)
    extract_depth_frame_embeddings(depth_frames_directory_path, depth_features_path)


if __name__ == "__main__":
    main()