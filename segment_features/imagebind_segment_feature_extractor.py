"""
We split the video file into fixed length segments
For incomplete segments, we ignore the remaining frames
We use these video segments to extract video embeddings using the imagebind model.
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType

from lib.imagebind.imagebind.data import SpatialCrop
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

# Load the model from checkpoint into the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def load_and_transform_video_data(
        video_path,
        clip_duration=2,
        sample_rate=16000,
):
    # Define transformations for the video
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=int(segment_length))
    video = EncodedVideo.from_path(
        video_path,
        decoder="decord",
        decode_audio=False
    )

    total_clips = video.duration // segment_length

    video_outputs = []
    for i in range(int(total_clips)):
        start_sample = i * segment_length
        end_sample = start_sample + segment_length
        clip = video.get_clip(start_sample, end_sample)
        if clip is None:
            raise ValueError("No clip found")
        # Apply transformations to the video clip
        video_clip = frame_sampler(clip['video'])
        video_clip = video_clip / 255.0  # since this is float, need 0-1
        transformed_clip = [video_transform(video_clip)]
        transformed_clip = SpatialCrop(224, num_crops=3)(transformed_clip)
        video_clip = torch.stack(transformed_clip, dim=0)
        video_outputs.append(video_clip)
    transformed_video = torch.stack(video_outputs, dim=0).to(device)
    return transformed_video


def fetch_video_embeddings(video_path):
    transformed_video = load_and_transform_video_data(video_path)

    # Initialize a list to hold the outputs
    output_embeddings = []

    # Process the video data in chunks of 100 using tqdm for progress visualization
    with torch.no_grad():
        for i in tqdm(range(0, len(transformed_video), 100), desc="Processing video chunks"):
            chunk = transformed_video[i:i + 100]  # Get the current chunk
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
        video_embeddings = fetch_video_embeddings(video_path)
        numpy_video_embeddings = video_embeddings.cpu().numpy()
        # Store embeddings in a npz file
        np.savez(npz_file_path, video_embeddings=numpy_video_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_length", type=float, required=True)

    segment_length = parser.parse_args().segment_length
    main()