# importing necessary libraries
import os
import sys
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from threading import Thread
import queue
from PIL import Image

class TemporalShift(nn.Module):
    #torch.Size([1, 8, 3, 224, 224])
    def __init__(self, n_segment, shift_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.shift_div = shift_div

    def forward(self, x):
        # x size: [N*T, C, H, W]
        N, T, C, H, W = x.size()
        x = x.view(N, T, C, H * W)

        # Zero padding for temporal dimension
        zero_pad = torch.zeros((N, 1, C, H * W), device=x.device, dtype=x.dtype)

        # Shift operation
        x = torch.cat((x[:, :-1], zero_pad), 1)  # Shift forward

        # Use clone() to avoid in-place operation issues
        x_temp = x.clone()
        x[:, 1:, :self.shift_div] = x_temp[:, :-1, :self.shift_div]  # Shift backward for some channels

        x = x.view(N, T, C, H, W)

        return x


class TSMFeatureExtractor(nn.Module):
    def __init__(self, n_segment):
        super(TSMFeatureExtractor, self).__init__()
        self.n_segment = n_segment

        # ResNet-101 Neural Network for feature extraction
        network = models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V1)
        modules = list(network.children())[:-2]  # Remove the last fully connected layer and avgpool
        self.resnet101 = nn.Sequential(*modules)
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

        return combined_features

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


if __name__ == '__main__':
    n_segment = 8  # This should match the number used in TSMFeatureExtractor
    tsm_features = TSMFeatureExtractor(n_segment)

    dir_path = sys.argv[1]
    video_features = {}
    frame_queue = queue.Queue()

    def process_batch():
        while True:
            batch_info = frame_queue.get()
            if batch_info is None:
                break

            root, frames_batch = batch_info
            processed_frames = []
            for file in frames_batch:
                frame_path = os.path.join(root, file)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = tsm_features.data_preprocessing(frame)
                processed_frames.append(frame)

            frame_tensor = torch.stack(processed_frames)
            frame_tensor = frame_tensor.unsqueeze(0)  # Add an extra dimension for batch

            extracted_features = tsm_features(frame_tensor)
            #video_features[os.path.join(root, str(frames_batch[0]))] = extracted_features
            frame_queue.task_done()

    # Create a pool of worker threads
    num_worker_threads = 4 
    threads = []
    for _ in range(num_worker_threads):
        t = Thread(target=process_batch)
        t.start()
        threads.append(t)

    for root, dirs, files in os.walk(dir_path):
        files.sort()  # Ensure the frames are sorted in the correct order
        for i in range(0, len(files), n_segment):
            frames_batch = files[i:i + n_segment]
            if len(frames_batch) != n_segment:
                continue  # Skip if the last batch does not have enough frames
            frame_queue.put((root, frames_batch))

    # Block until all tasks are done
    frame_queue.join()

    # Stop the worker threads
    for _ in range(num_worker_threads):
        frame_queue.put(None)
    for t in threads:
        t.join()