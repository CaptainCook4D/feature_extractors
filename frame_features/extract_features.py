import argparse
import datetime
import os
import numpy as np
import torch
import torchvision.transforms as T
import concurrent.futures
import cv2
import logging
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from omnivore_transforms import SpatialCrop, TemporalCrop
from PIL import Image
from natsort import natsorted


log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="omnivore", help="Specify the method to be used.")
    return parser.parse_args()


# Video Processing
class ImageProcessor:
    def __init__(self, method, feature_extractor, image_transform):
        self.method = method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor
        self.image_transform = image_transform

    def process_video(self, video_name, video_frames_directory_parent_path, output_features_path):
        video_frames_directory_path = os.path.join(video_frames_directory_parent_path, video_name)

        batch_size = 600
        output_file_path = os.path.join(output_features_path, video_name)
        os.makedirs(output_features_path, exist_ok=True)

        frames_list = sorted(os.listdir(video_frames_directory_path), key=lambda x: int(x.split("_")[1][:-4]))
        video_features = []
        for i in tqdm(range(0, len(frames_list), batch_size), desc=f"Extracting Features for images in video {video_name}"):
            batch_images = []
            batch_names = []
            # Load and preprocess images for the batch
            for frame_name in frames_list[i:i + batch_size]:
                image_path = os.path.join(video_frames_directory_path, frame_name)
                img = Image.open(image_path)
                input_tensor = self.image_transform(img).to(self.device)
                batch_images.append(input_tensor)
                batch_names.append(frame_name)

            # Stack the batch of images
            batch_images = torch.stack(batch_images).squeeze(dim=1)
            batch_images = batch_images.unsqueeze(dim=2)

            # Perform predictions for the batch
            batch_features = extract_features(
                batched_image_data=batch_images,
                feature_extractor=self.feature_extractor,
                method=self.method,
            )

            # Save the batch of predictions
            video_features.append(batch_features)

        video_features = np.vstack(video_features)
        np.savez(f"{output_file_path}.npz", video_features)
        logger.info(f"Saved features for video {video_name} at {output_file_path}.npz")


# Feature Extraction
def extract_features(batched_image_data, feature_extractor, method):
    with torch.no_grad():
        features = feature_extractor(batched_image_data)
    return features.cpu().numpy()


def get_image_transformation(model_name):
    if model_name == "omnivore":
        image_transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return image_transform


def get_feature_extractor(name, device="cuda"):
    if name == "omnivore":
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
        model.heads = torch.nn.Identity()

    feature_extractor = model
    feature_extractor = feature_extractor.to(device)
    feature_extractor = feature_extractor.eval()
    return feature_extractor


# Main
def main():
    args = parse_arguments()
    method = args.backbone

    if method is None:
        method = "omnivore"

    video_frames_directories_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p"
    output_features_path = f"/data/rohith/captain_cook/features/gopro/{method}/"

    image_transform = get_image_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = ImageProcessor(method, feature_extractor, image_transform)

    video_frame_directories = [file for file in os.listdir(video_frames_directories_path)]

    num_threads = 1
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    lambda video_frame_directory: processor.process_video(video_frame_directory, video_frames_directories_path, output_features_path), video_frame_directories
                ), total=len(video_frame_directories)
            )
        )


if __name__ == "__main__":
    main()
