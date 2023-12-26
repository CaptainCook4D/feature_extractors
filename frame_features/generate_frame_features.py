# importing necessary libraries
from collections import defaultdict
import os
import argparse
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from threading import Thread, Lock
from queue import Queue
import logging
from PIL import Image
from sklearn.decomposition import PCA
import pickle as pkl
import glob2 as glob

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)

feature_lock = Lock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        #self.pca_2048 = None
        for param in self.resnet101.parameters():
            param.requires_grad = False
        self.resnet101 = self.resnet101.to(device)  # Move the model to GPU
        self.tsm = TemporalShift(n_segment).to(device)

    def forward(self, x):
        # Step 1: Feature extraction with ResNet-101
        x = x.to(device)
        N, T, C, H, W = x.size()
        x = x.view(N * T, C, H, W)
        original_features = self.resnet101(x)

        # Reshape back to add the temporal dimension
        original_features = original_features.view(N, T, -1, H, W)

        # Step 2: Temporal Shifting
        shifted_features = self.tsm(original_features)

        # Step 3: Combining the original and shifted features
        combined_features = torch.cat((original_features, shifted_features), dim=2)

        flattened = combined_features.view(combined_features.size(0), -1)

        # Linear layer to get to the desired feature size of 2048
        fc = torch.nn.Linear(in_features=flattened.size(1), out_features=2048)

        # Reshape to (n, 2048)
        frame_features = fc(flattened)

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
        return preprocess(frame).to(device)

def total_files(video_name):
    pattern = '*.jpg'
    jpg_files = glob.glob(video_name + "/" + pattern)
    return len(jpg_files)

def process_batch(video_name, root, frames_batch, feature_map):
    processed_frames = []
    for file in frames_batch:
        frame_path = os.path.join(root, file)
        frame = Image.open(frame_path)
        frame = tsm_features.data_preprocessing(frame)
        processed_frames.append(frame)

    if processed_frames:
        frame_tensor = torch.stack(processed_frames)
        frame_tensor = frame_tensor.unsqueeze(0)

        frame_tensor = frame_tensor.to(device)
        extracted_features = tsm_features(frame_tensor)
        if isinstance(extracted_features, torch.Tensor):
            extracted_features_np = extracted_features.cpu().detach().numpy()
        else:
            extracted_features_np = extracted_features

        feature_map.append(extracted_features_np)
    
    with feature_lock:
        feature_map[video_name].append(extracted_features_np)
    
    return feature_map
    

def worker(queue, feature_map):
    while True:
        task = queue.get()
        if task is None:
            break
        video_name, root, frames_batch = task
        try:
            process_batch(video_name, root, frames_batch, feature_map)
        except BaseException as e:
            print("An error occurred while processing:", e)
        finally:
            queue.task_done()

def main(n_segment, video_frames_directories_path, output_features_path, batch_size=None):
    num_worker_threads = 1

    # Create the queue and the worker threads
    queue = Queue()

    threads = []
    feature_map = defaultdict(list)  # Changed to a dictionary to hold features for each video
    
    for _ in range(num_worker_threads):
        t = Thread(target=worker, args=(queue, feature_map))
        t.start()
        threads.append(t)

    try:
        if os.path.getsize(output_features_path) > 0:
            for root, dirs, files in os.walk(video_frames_directories_path):
                video_name = os.path.basename(root)

                files.sort()
                total_frames = len(files)
                if batch_size is None:
                    batch_size = n_segment  # default batch size

                # Ensure each batch has an equal number of frames
                print(f"Extracting features from: {video_name}")
                for i in range(0, total_frames, batch_size):
                    frames_batch = files[i:i + batch_size]
                    if frames_batch:
                        queue.put((video_name, root, frames_batch))
                
                for video_name, features in feature_map.items():
                    feature_file_path = os.path.join(output_features_path, video_name)
                    np.savez(f"{feature_file_path}.npz", *features)
                    logger.info(f"Saved features for video {video_name} at {feature_file_path}.npz")

    except BaseException as e:
        print("An error occurred:", e)

    # Block until all tasks are done
    queue.join()

    # Stop workers
    for i in range(num_worker_threads):
        queue.put(None)
    for t in threads:
        t.join()

if __name__ == '__main__':
    n_segment = 8
    tsm_features = TSMFeatureExtractor(n_segment).to(device)
    args = parse_arguments()
    method = args.backbone or "tsm"

    video_frames_directories_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p/10_16_360p"

    output_features_path = f"/data/rohith/captain_cook/features/gopro/frames/{method}/"
    os.makedirs(output_features_path, exist_ok=True)

    total_frames = total_files(video_frames_directories_path)
    desired_num_batches = 25
    batch_size = max(total_frames // desired_num_batches, 1)

    main(n_segment, video_frames_directories_path, output_features_path, batch_size)



