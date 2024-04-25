import argparse
import concurrent.futures
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.25)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="tsm", help="Specify the method to be used.")
    parser.add_argument("--batch", type=str, default=None, help="Specify the batch of videos to extract features from")
    return parser.parse_args()


class TSMFeatureExtractor():
    def __init__(self, n_segment):
        super(TSMFeatureExtractor, self).__init__()
        self.n_segment = n_segment
        network = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        modules = list(network.children())[:-2]
        self.resnet101 = nn.Sequential(*modules)
        for param in self.resnet101.parameters():
            param.requires_grad = False
        self.resnet101 = self.resnet101.to(device)

    @staticmethod
    def temporal_shift(x):
        N, T, C, H, W = x.size()
        x_new = x.view(N * T, C, H, W)
        zero_pad = torch.zeros((N, 1, C, H, W), device=x.device, dtype=x.dtype)

        # Adding a zero frame for shifting
        x = torch.cat((x, zero_pad), 1)

        shift_div = C // 4

        out = torch.zeros_like(x)
        out[:, :-1, :shift_div] = x[:, 1:, :shift_div]  # shift left
        out[:, 1:, shift_div: 2 * shift_div] = x[:, :-1, shift_div: 2 * shift_div]  # shift right
        out[:, :, 2 * shift_div:] = x[:, :, 2 * shift_div:]  # no shift

        out = out[:, 1:, :, :, :]

        out = out.view(N, T, C, H, W)

        return out

    def tsm_features(self, x):
        x = x.to(device)
        N, T, C, H, W = x.size()

        shifted_features = self.temporal_shift(x)
        shifted_features = shifted_features.view(N * T, C, H, W)

        features = self.resnet101(shifted_features)

        flattened = features.view(features.size(0), -1)
        fc = torch.nn.Linear(in_features=flattened.size(1), out_features=2048)

        frame_features = fc(flattened)

        return frame_features


class Processor():
    @staticmethod
    def frame_processing(frame):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(frame).to(device)

    @staticmethod
    def process_batch(batch_frames, tsm_extractor):
        '''
            i/p: batch of 1000 frames, tsm_extractor object
        
            o/p: tsm_extractor output is an array of shape [8, 2048]
            final o/p size is [16384, ]
        '''
        try:
            batch_features = []
            n_segment = 8
            for i in range(0, len(batch_frames), n_segment):
                segment_frames = batch_frames[i:i + n_segment]

                segment_frames = torch.stack(segment_frames)
                segment_frames = segment_frames.unsqueeze(dim=0)

                extracted_features = tsm_extractor.tsm_features(segment_frames)
                if isinstance(extracted_features, torch.Tensor):
                    extracted_features_np = extracted_features.cpu().detach().numpy()
                else:
                    extracted_features_np = extracted_features

                batch_features.append(extracted_features_np)

            return batch_features

        except BaseException as e:
            print("Error in execution of process_batch: ", e)

    @staticmethod
    def process_video(video_name, video_frames_directories_path, output_features_path, tsm_extractor):
        '''
            i/p: video name, directory of videos, output path, tsm feature extractor method

            o/p: .npz file containing frame wise features of each video in batches of 1000
        '''
        try:
            video_directory = os.path.join(video_frames_directories_path, video_name)
            feature_path = os.path.join(output_features_path, video_name)
            frames = sorted(os.listdir(video_directory), key=lambda x: int(x.split("_")[1][:-4]))
            batch_size = 2096
            video_features = []
            for i in tqdm(range(0, len(frames), batch_size), desc=f"TSM Feature Extraction for video: {video_name}"):
                batch_frames = []
                batch_names = []
                for frame in frames[i:i + batch_size]:
                    frame_path = os.path.join(video_directory, frame)
                    image = Image.open(frame_path)
                    image = Processor.frame_processing(image)
                    batch_frames.append(image)
                    batch_names.append(frame)

                batch_features = Processor.process_batch(batch_frames, tsm_extractor)

                video_features.extend(batch_features)

            video_features = np.vstack(video_features)
            np.savez(f"{feature_path}.npz", video_features)
            logger.info(f"Saved featured for video {video_name} at {feature_path}")
            print("\n")

        except BaseException as e:
            print("Error in execution of process_video: ", e)

        return


def main():
    n_segment = 8
    args = parse_arguments()
    tsm_features = TSMFeatureExtractor(n_segment)
    method = args.backbone or "tsm"

    video_frames_directories_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p/"

    output_features_path = f"/data/rohith/captain_cook/features/gopro/frames/{method}/"

    #completed_videos = [folder.split(".")[0] for folder in os.listdir(output_features_path)]

    num_worker_threads = 1
    processor = Processor()

    try:
        batch_videos = args.batch
        video_folders = batch_videos.split(',')

        #print(completed_videos)
        #print(video_folders)

        with concurrent.futures.ThreadPoolExecutor(num_worker_threads) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda video_name: processor.process_video(video_name,
                                                                   video_frames_directories_path=video_frames_directories_path,
                                                                   output_features_path=output_features_path,
                                                                   tsm_extractor=tsm_features), video_folders
                    ), total=len(video_folders)
                )
            )
            print("\n")

    except BaseException as e:
        print("Error occurred in execution: ", e)


if __name__ == '__main__':
    main()
