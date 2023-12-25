# importing necessary libraries
import os
import argparse
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from threading import Thread
import queue
import logging
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
import pickle as pkl

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="tsm", help="Specify the method to be used.")
    return parser.parse_args()


class TemporalShift(nn.Module):
    # torch.Size([1, 8, 3, 224, 224])
    def __init__(self, n_segment, shift_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.shift_div = shift_div

    def forward(self, x):
        N, T, C, H, W = x.size()
        x = x.view(N, T, C, H * W)

        # Zero padding for temporal dimension
        zero_pad = torch.zeros((N, 1, C, H * W), device=x.device, dtype=x.dtype)

        # Shift forward
        x = torch.cat((x[:, :-1], zero_pad), 1)

        # Shift backward for some channels
        x_temp = x.clone()
        x[:, 1:, :self.shift_div] = x_temp[:, :-1, :self.shift_div]

        x = x.view(N, T, C, H, W)

        return x


class TSMFeatureExtractor(nn.Module):
    def __init__(self, n_segment):
        super(TSMFeatureExtractor, self).__init__()
        self.n_segment = n_segment

        # ResNet-101 Neural Network for feature extraction
        network = models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer and avg pooling layer
        modules = list(network.children())[:-2]
        self.resnet101 = nn.Sequential(*modules)
        self.pca_2048 = PCA(n_components=2048)
        for param in self.resnet101.parameters():
            param.requires_grad = False

        self.tsm = TemporalShift(n_segment)

    def forward(self, x):
        # Step 1: Feature extraction with ResNet-101
        N, T, C, H, W = x.size()
        x = x.view(N * T, C, H, W)
        original_features = self.resnet101(x)

        # Reshape back to add the temporal dimension
        original_features = original_features.view(N, T, -1, H, W)

        # Step 2: Temporal Shifting
        shifted_features = self.tsm(original_features)

        # Step 3: Combining the original and shifted features
        combined_features = torch.cat((original_features, shifted_features), dim=2)

        # Reshape for final output
        N, T, C, H, W = combined_features.size()
        combined_features = combined_features.view(N * T, C, H, W)

        flattened_features = combined_features.reshape(-1)
        frame_features = self.pca_2048.fit_transform(flattened_features)

        return frame_features

    @staticmethod
    def data_preprocessing(frame):
        '''Preprocess the frame'''
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(frame)

#load_checkpoint to handle a dictionary of frames
def load_checkpoint(video_name):
    '''Load the last checkpoint if it exists'''
    checkpoint_path = os.path.join(output_features_path, video_name, 'checkpoint.pkl')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pkl.load(f)
    return {}

#save_checkpoint to store frame names in a dictionary
def save_checkpoint(video_name, frames_batch):
    '''Save the current state to continue later'''
    checkpoint_path = os.path.join(output_features_path, video_name, 'checkpoint.pkl')
    finished_frames = load_checkpoint(video_name)
    for frame in frames_batch:
        finished_frames[frame] = True
    with open(checkpoint_path, 'wb') as f:
        pkl.dump(finished_frames, f)

def process_batch(video_name, root, frames_batch, finished_frames):
    video_folder_path = os.path.join(output_features_path, video_name)
    os.makedirs(video_folder_path, exist_ok=True)

    processed_frames = []
    for file in frames_batch:
        # Skip if frame is already processed
        if finished_frames.get(file):
            continue

        frame_path = os.path.join(root, file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = tsm_features.data_preprocessing(frame)
        processed_frames.append(frame)

    if processed_frames:
        frame_tensor = torch.stack(processed_frames)
        frame_tensor = frame_tensor.unsqueeze(0)

        extracted_features = tsm_features(frame_tensor)
        extracted_features_np = extracted_features.cpu().detach().numpy()
        feature_file_path = os.path.join(video_folder_path, f"{frames_batch[0]}.npz")
        np.savez(feature_file_path, extracted_features_np)
        save_checkpoint(video_name, frames_batch)

if __name__ == '__main__':
    n_segment = 8
    tsm_features = TSMFeatureExtractor(n_segment)
    args = parse_arguments()
    method = args.backbone or "tsm"

    video_frames_directories_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p"
    output_features_path = f"/data/rohith/captain_cook/features/gopro/frames/{method}/"

    # ThreadPoolExecutor setup
    num_worker_threads = 4
    try:
        with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
            futures = []
            for root, dirs, files in os.walk(video_frames_directories_path):
                video_name = os.path.basename(root)
                finished_frames = load_checkpoint(video_name)

                print("Extracting features for " + video_name)
                files.sort()
                for i in range(0, len(files), n_segment):
                    frames_batch = files[i:i + n_segment]
                    if len(frames_batch) == n_segment:
                        futures.append(executor.submit(process_batch, video_name, root, frames_batch, finished_frames))

            for future in as_completed(futures):
                future.result()
    except BaseException as e:  # Catching the exception
        print("An error occurred:", e)


