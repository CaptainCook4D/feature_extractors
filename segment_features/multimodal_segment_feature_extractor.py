import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from lib.imagebind.imagebind import ModalityType
from lib.imagebind.imagebind.models import imagebind_model

# Load the model from checkpoint into the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = imagebind_model.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)

# Custom Dataset for RGB images
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
        image = self.transform(image)
        return image

# Custom Dataset for Depth images
class DepthDataset(Dataset):
    def __init__(self, depth_paths):
        self.depth_paths = depth_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.depth_paths)

    def __getitem__(self, idx):
        depth_path = self.depth_paths[idx]
        with open(depth_path, "rb") as fopen:
            image = Image.open(fopen).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        disparity = Image.fromarray(image)
        image = self.transform(disparity)
        return image

def extract_rgb_frame_embeddings_for_recording(recording_frames_path):
    print("Loading and transforming RGB data")
    dataset = ImageDataset(recording_frames_path)
    dataloader = DataLoader(dataset, batch_size=30, num_workers=8, pin_memory=True)

    output_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing RGB batches"):
            batch = batch.to(device)
            inputs = {ModalityType.VISION: batch}
            embeddings = model(inputs)
            # Aggregate embeddings by taking the mean over the batch
            aggregated_embedding = torch.mean(embeddings[ModalityType.VISION], dim=0, keepdim=True)
            output_embeddings.append(aggregated_embedding.cpu())

    # Stack all the collected embeddings into a single tensor
    stacked_embeddings = torch.cat(output_embeddings, dim=0)
    return stacked_embeddings

def extract_depth_frame_embeddings_for_recording(recording_frames_path):
    print("Loading and transforming Depth data")
    dataset = DepthDataset(recording_frames_path)
    dataloader = DataLoader(dataset, batch_size=30, num_workers=8, pin_memory=True)

    output_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Depth batches"):
            batch = batch.to(device)
            inputs = {ModalityType.DEPTH: batch}
            embeddings = model(inputs)
            # Aggregate embeddings by taking the mean over the batch
            aggregated_embedding = torch.mean(embeddings[ModalityType.DEPTH], dim=0, keepdim=True)
            output_embeddings.append(aggregated_embedding.cpu())

    # Stack all the collected embeddings into a single tensor
    stacked_embeddings = torch.cat(output_embeddings, dim=0)
    return stacked_embeddings

def extract_rgb_frame_embeddings(rgb_frames_directory_path, rgb_features_path):
    rgb_files = os.listdir(rgb_frames_directory_path)
    for rgb_file in tqdm(rgb_files, desc="Processing RGB files"):
        print(f"\nProcessing RGB file: {rgb_file}")
        rgb_file_path = os.path.join(rgb_frames_directory_path, rgb_file)
        rgb_frames_path_list = [os.path.join(rgb_file_path, f) for f in os.listdir(rgb_file_path)]
        rgb_frame_embeddings = extract_rgb_frame_embeddings_for_recording(rgb_frames_path_list)
        rgb_features_file_path = os.path.join(rgb_features_path, f"{rgb_file[:-5]}.npz")
        np.savez(rgb_features_file_path, embeddings=rgb_frame_embeddings.numpy())

def extract_depth_frame_embeddings(depth_frames_directory_path, depth_features_path):
    depth_files = os.listdir(depth_frames_directory_path)
    for depth_file in tqdm(depth_files, desc="Processing Depth files"):
        print(f"\nProcessing Depth file: {depth_file}")
        depth_file_path = os.path.join(depth_frames_directory_path, depth_file)
        depth_frames_paths_list = [os.path.join(depth_file_path, f) for f in os.listdir(depth_file_path)]
        depth_frame_embeddings = extract_depth_frame_embeddings_for_recording(depth_frames_paths_list)
        depth_features_file_path = os.path.join(depth_features_path, f"{depth_file[:-5]}.npz")
        np.savez(depth_features_file_path, embeddings=depth_frame_embeddings.numpy())

def main(mode):
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

    if mode == "rgb":
        extract_rgb_frame_embeddings(rgb_frames_directory_path, rgb_features_path)
    elif mode == "depth":
        extract_depth_frame_embeddings(depth_frames_directory_path, depth_features_path)

def segment_feature_generator(segment_length):
    narration_feature_directory_path = f"/data/rohith/captain_cook/features/gopro/segments/depth/"
    segment_narration_feature_directory_path = f"/data/rohith/captain_cook/features/gopro/segments_{segment_length}/depth/"
    os.makedirs(segment_narration_feature_directory_path, exist_ok=True)

    npz_files = [f for f in os.listdir(narration_feature_directory_path) if f.endswith('.npz')]

    for npz_file in tqdm(npz_files, desc="Processing files"):
        npz_file_path = os.path.join(narration_feature_directory_path, npz_file)
        segment_npz_file_path = os.path.join(segment_narration_feature_directory_path, npz_file)

        if os.path.exists(segment_npz_file_path):
            print(f"Skipping {npz_file}")
            continue

        # Load the npz file
        with np.load(npz_file_path) as data:
            video_embeddings = data['embeddings']

        # Calculate the number of segments
        num_segments = math.ceil(video_embeddings.shape[0] / segment_length)
        # num_segments = video_embeddings.shape[0] // segment_length

        # Initialize a list to hold the segment embeddings
        segment_embeddings = []

        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment_embedding = video_embeddings[start:end].mean(axis=0)
            segment_embeddings.append(segment_embedding)

        # Stack all the segment embeddings into a single tensor
        stacked_segment_embeddings = np.stack(segment_embeddings)

        # Store embeddings in a npz file
        np.savez(segment_npz_file_path, video_embeddings=stacked_segment_embeddings)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, required=True)
    # args = parser.parse_args()
    # main(args.mode)
    segment_feature_generator(segment_length=2)
