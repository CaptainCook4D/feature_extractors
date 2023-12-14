import argparse
import datetime
import glob
import os
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
	ApplyTransformToKey,
	ShortSideScale,
	UniformTemporalSubsample
)
import torchvision.transforms as T
import concurrent.futures
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
	parser.add_argument("--backbone", type=str, required=True, help="Specify the method to be used.")
	return parser.parse_args()


# Video Processing
class VideoProcessor:
	def __init__(self, method, feature_extractor, video_transform):
		self.method = method
		self.feature_extractor = feature_extractor
		self.video_transform = video_transform
		
		self.fps = 30
		self.num_frames_per_feature = 30
	
	def process_video(self, video_name, video_directory_path, output_features_path):
		segment_size = self.fps / self.num_frames_per_feature
		video_path = os.path.join(video_directory_path, f"{video_name}.mp4")
		
		output_file_path = os.path.join(output_features_path, video_name)
		os.makedirs(output_features_path, exist_ok=True)
		
		video = EncodedVideo.from_path(video_path)
		video_duration = video.duration
		
		logger.info(f"video: {video_name} video_duration: {video_duration} s")
		segment_end = max(video_duration - segment_size + 1, 1)
		
		video_features = []
		for start_time in tqdm(np.arange(0, segment_end, segment_size),
		                       desc="Processing video segments for video {video_name}"):
			end_time = start_time + segment_size
			end_time = min(end_time, video_duration)
			
			video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
			segment_video_inputs = video_data["video"]
			
			segment_features = extract_features(
				video_data_raw=segment_video_inputs,
				feature_extractor=self.feature_extractor,
				transforms_to_apply=self.video_transform,
				method=self.method
			)
			
			video_features.append(segment_features)
		
		video_features = np.vstack(video_features)
		np.savez(f"{output_file_path}_{segment_size}s.npz", video_features)


# Feature Extraction
def extract_features(video_data_raw, feature_extractor, transforms_to_apply, method):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	video_data_for_transform = {"video": video_data_raw, "audio": None}
	video_data = transforms_to_apply(video_data_for_transform)
	video_inputs = video_data["video"]
	if method in ["omnivore"]:
		video_input = video_inputs[0][None, ...].to(device)
	elif method == "slowfast":
		video_input = [i.to(device)[None, ...] for i in video_inputs]
	elif method == "x3d_pca_nc64":
		video_input = video_inputs.unsqueeze(0).to(device)
	elif method == "3dresnet":
		video_input = video_inputs.unsqueeze(0).to(device)
	with torch.no_grad():
		features = feature_extractor(video_input)
	return features.cpu().numpy()


# Model Initialization
def get_video_transformation(name):
	if name == "omnivore":
		num_frames = 32
		video_transform = T.Compose(
			[
				UniformTemporalSubsample(num_frames),
				T.Lambda(lambda x: x / 255.0),
				ShortSideScale(size=224),
				NormalizeVideo(
					mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
				),
				TemporalCrop(frames_per_clip=32, stride=40),
				SpatialCrop(crop_size=224, num_crops=3),
			]
		)
	elif name == "slowfast":
		slowfast_alpha = 4
		num_frames = 32
		side_size = 256
		crop_size = 256
		mean = [0.45, 0.45, 0.45]
		std = [0.225, 0.225, 0.225]
		
		class PackPathway(torch.nn.Module):
			def __init__(self):
				super().__init__()
			
			def forward(self, frames: torch.Tensor):
				fast_pathway = frames
				# Perform temporal sampling from the fast pathway.
				slow_pathway = torch.index_select(
					frames,
					1,
					torch.linspace(
						0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
					).long(),
				)
				frame_list = [slow_pathway, fast_pathway]
				return frame_list
		
		video_transform = T.Compose(
			[
				UniformTemporalSubsample(num_frames),
				Lambda(lambda x: x / 255.0),
				NormalizeVideo(mean, std),
				ShortSideScale(size=side_size),
				CenterCropVideo(crop_size),
				PackPathway(),
			]
		)
	elif name == "x3d_pca_nc64":
		mean = [0.45, 0.45, 0.45]
		std = [0.225, 0.225, 0.225]
		model_transform_params = {
			"x3d_xs": {
				"side_size": 182,
				"crop_size": 182,
				"num_frames": 4,
				"sampling_rate": 12,
			},
			"x3d_s": {
				"side_size": 182,
				"crop_size": 182,
				"num_frames": 13,
				"sampling_rate": 6,
			},
			"x3d_m": {
				"side_size": 256,
				"crop_size": 256,
				"num_frames": 16,
				"sampling_rate": 5,
			},
		}
		# Taking x3d_m as the model
		transform_params = model_transform_params["x3d_m"]
		video_transform = Compose(
			[
				UniformTemporalSubsample(transform_params["num_frames"]),
				Lambda(lambda x: x / 255.0),
				NormalizeVideo(mean, std),
				ShortSideScale(size=transform_params["side_size"]),
				CenterCropVideo(
					crop_size=(
						transform_params["crop_size"],
						transform_params["crop_size"],
					)
				),
			]
		)
	elif name == "3dresnet":
		side_size = 256
		mean = [0.45, 0.45, 0.45]
		std = [0.225, 0.225, 0.225]
		crop_size = 256
		num_frames = 8
		video_transform = Compose(
			[
				UniformTemporalSubsample(num_frames),
				Lambda(lambda x: x / 255.0),
				NormalizeVideo(mean, std),
				ShortSideScale(size=side_size),
				CenterCropVideo(crop_size=(crop_size, crop_size)),
			]
		)
	return ApplyTransformToKey(key="video", transform=video_transform)


def get_feature_extractor(name, device="cuda"):
	if name == "omnivore":
		model_name = "omnivore_swinB_epic"
		model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
		model.heads = torch.nn.Identity()
	elif name == "slowfast":
		model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
		model.heads = torch.nn.Identity()
	elif name == "x3d":
		model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
		model.heads = torch.nn.Identity()
	elif name == "3dresnet":
		model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
		model.heads = torch.nn.Identity()
	
	feature_extractor = model
	feature_extractor = feature_extractor.to(device)
	feature_extractor = feature_extractor.eval()
	return feature_extractor


def main_hololens():
	args = parse_arguments()
	method = args.backbone
	
	hololens_directory_path = "/data/rohith/captain_cook/data/hololens/"
	output_features_path = f"/data/rohith/captain_cook/features/hololens/segments/{method}/"
	
	video_transform = get_video_transformation(method)
	feature_extractor = get_feature_extractor(method)
	
	processor = VideoProcessor(method, feature_extractor, video_transform)
	
	num_threads = 10
	with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
		for recording_id in os.listdir(hololens_directory_path):
			video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
			executor.submit(processor.process_video, recording_id, video_file_path, output_features_path)


# Main
def main():
	args = parse_arguments()
	method = args.backbone
	
	video_files_path = "/data/error_detection/dataset/videos/recordings"
	output_features_path = f"/data/error_detection/dataset/features/{method}"
	
	video_transform = get_video_transformation(method)
	feature_extractor = get_feature_extractor(method)
	
	processor = VideoProcessor(method, feature_extractor, video_transform)
	
	mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]
	
	num_threads = 10
	with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
		list(
			tqdm(
				executor.map(
					lambda file: processor.process_video(file, video_files_path, output_features_path), mp4_files
				), total=len(mp4_files)
			)
		)


if __name__ == "__main__":
	main_hololens()
