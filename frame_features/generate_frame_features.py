#import libraries
from collections import defaultdict
import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import logging
from PIL import Image
import glob2 as glob
import concurrent.futures
import tqdm

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.5)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="tsm", help="Specify the method to be used.")
    return parser.parse_args()

class TSMFeatureExtractor():
    def __init__(self, n_segment):
        super(TSMFeatureExtractor, self).__init__()
        self.n_segment = n_segment
        network = models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V1)
        modules = list(network.children())[:-2]
        self.resnet101 = nn.Sequential(*modules)
        for param in self.resnet101.parameters():
            param.requires_grad = False
        self.resnet101 = self.resnet101.to(device)

    @staticmethod
    def temporal_shift(x):
        N, T, C, H, W = x.size()
        x = x.view(N, T, C, H*W)
        zero_pad = torch.zeros((N, 1, C, H * W), device = x.device, dtype = x.dtype)
        x = torch.cat((x[:,:-1], zero_pad), 1)

        shift_div = C // 4

        out = torch.zeros_like(x)
        out[:, :-1, :shift_div] = x[:, 1:, :shift_div]  # shift left
        out[:, 1:, shift_div: 2 * shift_div] = x[:, :-1, shift_div: 2 * shift_div]  # shift right
        out[:, :, 2 * shift_div:] = x[:, :, 2 * shift_div:]

        out = out.view(N, T, C, H, W)

        return out

    def tsm_features(self, x):
        x = x.to(device)
        N, T, C, H, W = x.size()

        x = x.view(N * T, C, H, W)

        shifted_features = self.temporal_shift(x)
        features = self.resnet101(shifted_features)

        flattened = features.view(features.size(0), -1)
        fc = torch.nn.Linear(in_features = flattened.size(1), out_features=2048)

        frame_features = fc(flattened)

        return frame_features

class Processor():
    def __init__(self, tsm_features):
        self.tsm_extractor = tsm_features

    def frame_processing(frame):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
        return preprocess(frame).to(device)

    def process_batch(self,batch_frames):
        batch_features = []
        n_segment = 8
        for i in range(0, len(batch_frames), n_segment):
            frame = batch_frames[i:i+n_segment]
            extracted_features = self.tsm_extractor(frame)
            if isinstance(extracted_features, torch.Tensor):
                extracted_features_np = extracted_features.cpu().detach().numpy()
            else:
                extracted_features_np = extracted_features

            extracted_features_np = extracted_features_np.flatten()

            batch_features.append(extracted_features_np)
        
        return batch_features

    @staticmethod
    def process_video(video_name, video_frames_directories_path, output_features_path):
        video_directory = os.path.join(video_frames_directories_path, video_name)
        feature_path = os.path.join(output_features_path,  video_name)
        frames = sorted(os.listdir(video_directory), key=lambda x: int(x.split("_")[1][:-4]))
        batch_size = 1000
        video_features = []
        for i in tqdm(range(0, len(frames), batch_size), desc=f"TSM Feature Extraction for video: {video_name}"):
            batch_frames = []
            batch_names = []
            for frame in frames[i:i+batch_size]:
                frame_path = os.path.join(video_directory, frame)
                image = Image.open(frame_path)
                image = Processor.frame_processing(image)
                batch_frames.append(image)
                batch_names.append(frame)

            batch_frames = torch.stack(batch_frames).squeeze(dim=1)
            batch_frames = batch_frames.unsqueeze(dim=2)

            batch_features = Processor.process_batch(batch_frames)

            video_features.append(batch_features)

        video_features = np.vstack(video_features)
        np.savez(f"{feature_path}.npz", video_features)
        logger.info(f"Saved featured for video {video_name} at {feature_path}")
        return

def main():
    n_segment = 8
    tsm_features = TSMFeatureExtractor(n_segment)
    args = parse_arguments()
    method = args.backbone or "tsm"

    video_frames_directories_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p/"

    output_features_path = f"/data/rohith/captain_cook/features/gopro/frames/{method}/"
    os.makedirs(output_features_path, exist_ok=True)

    num_worker_threads = 1
    processor = Processor(tsm_features)

    try:
        video_folders = [folder for folder in os.listdir(video_frames_directories_path)]

        with concurrent.futures.ThreadPoolExecutor(num_worker_threads) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda video_name: processor.process_video(video_name, video_frames_directories_path=video_frames_directories_path, output_features_path = output_features_path), video_folders
                    ), total=len(video_folders)
                )
            )

    except BaseException as e:
        print("Error occurred in execution: ", e)

if __name__ == '__main__':
    main()