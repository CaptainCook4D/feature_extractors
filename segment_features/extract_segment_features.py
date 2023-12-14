# import argparse
# import datetime
# import glob
# import time
# from natsort import natsorted
# import numpy as np
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Lambda
# from tqdm import tqdm
# import os
# from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample,
#     UniformCropVideo,
# )
# import concurrent.futures
# import torchvision.transforms as T
# import torch
# from omnivore_transforms import SpatialCrop, TemporalCrop
# from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
# from torchvision.models.video import (
#     mvit_v2_s,
#     MViT_V2_S_Weights,
# )
# from transformers import VivitImageProcessor, VivitModel
#
#
# def get_video_transformation(name):
#     # Define the specific transformations
#     if name == "omnivore":
#         num_frames = 32
#         video_transform = ApplyTransformToKey(
#             key="video",
#             transform=T.Compose(
#                 [
#                     UniformTemporalSubsample(num_frames),
#                     T.Lambda(lambda x: x / 255.0),
#                     ShortSideScale(size=224),
#                     NormalizeVideo(
#                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                     ),
#                     TemporalCrop(frames_per_clip=32, stride=40),
#                     SpatialCrop(crop_size=224, num_crops=3),
#                 ]
#             ),
#         )
#     elif name == "slowfast":
#         slowfast_alpha = 4
#         num_frames = 32
#         side_size = 256
#         crop_size = 256
#         mean = [0.45, 0.45, 0.45]
#         std = [0.225, 0.225, 0.225]
#
#         class PackPathway(torch.nn.Module):
#             """
#             Transform for converting video frames as a list of tensors.
#             """
#
#             def __init__(self):
#                 super().__init__()
#
#             def forward(self, frames: torch.Tensor):
#                 fast_pathway = frames
#                 # Perform temporal sampling from the fast pathway.
#                 slow_pathway = torch.index_select(
#                     frames,
#                     1,
#                     torch.linspace(
#                         0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
#                     ).long(),
#                 )
#                 frame_list = [slow_pathway, fast_pathway]
#                 return frame_list
#
#         video_transform = ApplyTransformToKey(
#             key="video",
#             transform=T.Compose(
#                 [
#                     UniformTemporalSubsample(num_frames),
#                     Lambda(lambda x: x / 255.0),
#                     NormalizeVideo(mean, std),
#                     ShortSideScale(size=side_size),
#                     CenterCropVideo(crop_size),
#                     PackPathway(),
#                 ]
#             ),
#         )
#     elif name == "x3d_pca_nc64":
#         mean = [0.45, 0.45, 0.45]
#         std = [0.225, 0.225, 0.225]
#         model_transform_params = {
#             "x3d_xs": {
#                 "side_size": 182,
#                 "crop_size": 182,
#                 "num_frames": 4,
#                 "sampling_rate": 12,
#             },
#             "x3d_s": {
#                 "side_size": 182,
#                 "crop_size": 182,
#                 "num_frames": 13,
#                 "sampling_rate": 6,
#             },
#             "x3d_m": {
#                 "side_size": 256,
#                 "crop_size": 256,
#                 "num_frames": 16,
#                 "sampling_rate": 5,
#             },
#         }
#         # Taking x3d_m as the model
#         transform_params = model_transform_params["x3d_m"]
#         video_transform = ApplyTransformToKey(
#             key="video",
#             transform=Compose(
#                 [
#                     UniformTemporalSubsample(transform_params["num_frames"]),
#                     Lambda(lambda x: x / 255.0),
#                     NormalizeVideo(mean, std),
#                     ShortSideScale(size=transform_params["side_size"]),
#                     CenterCropVideo(
#                         crop_size=(
#                             transform_params["crop_size"],
#                             transform_params["crop_size"],
#                         )
#                     ),
#                 ]
#             ),
#         )
#     elif name == "3dresnet":
#         side_size = 256
#         mean = [0.45, 0.45, 0.45]
#         std = [0.225, 0.225, 0.225]
#         crop_size = 256
#         num_frames = 8
#
#         # Note that this transform is specific to the slow_R50 model.
#         video_transform = ApplyTransformToKey(
#             key="video",
#             transform=Compose(
#                 [
#                     UniformTemporalSubsample(num_frames),
#                     Lambda(lambda x: x / 255.0),
#                     NormalizeVideo(mean, std),
#                     ShortSideScale(size=side_size),
#                     CenterCropVideo(crop_size=(crop_size, crop_size)),
#                 ]
#             ),
#         )
#     elif name == "mvit":
#         # Step 2: Initialize the inference transforms
#         weights = MViT_V2_S_Weights.DEFAULT
#         video_transform = weights.transforms()
#     # elif name == "swin":
#     #     weights = Swin3D_T_Weights.DEFAULT
#     #     video_transform = weights.transforms()
#
#     return video_transform
#
#
# def get_feature_extractor(name, device="cuda"):
#     if name == "omnivore":
#         model_name = "omnivore_swinB_epic"
#         model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
#         model.heads = torch.nn.Identity()
#         feature_extractor = model
#     elif name == "slowfast":
#         # Pick a pretrained model and load the pretrained weights
#         model = torch.hub.load(
#             "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True
#         )
#         model.heads = torch.nn.Identity()
#         # model.blocks = model.blocks[:-1]
#         feature_extractor = model
#     elif name == "r2plus1d_r50":
#         model = torch.hub.load("facebookresearch/pytorchvideo", name, pretrained=True)
#     elif name == "x3d_pca_nc64":
#         model_name = "x3d_m"
#         model = torch.hub.load(
#             "facebookresearch/pytorchvideo", model_name, pretrained=True
#         )
#         model.blocks = model.blocks[:-1]
#         feature_extractor = model
#     elif name == "x3d":
#         model_name = "x3d_m"
#         model = torch.hub.load(
#             "facebookresearch/pytorchvideo", model_name, pretrained=True
#         )
#         model.heads = torch.nn.Identity()
#         feature_extractor = model
#     elif name == "3dresnet":
#         model = torch.hub.load(
#             "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
#         )
#         # model.blocks = model.blocks[:-1]
#         model.heads = torch.nn.Identity()
#         feature_extractor = model
#     elif name == "mvit":
#         # model_name = "mvit_base_32x3"
#         # feature_extractor = torch.hub.load(
#         #     "facebookresearch/pytorchvideo", model_name, pretrained=True
#         # )
#         weights = MViT_V2_S_Weights.DEFAULT
#         feature_extractor = mvit_v2_s(weights=weights)
#         # feature_extractor.heads = torch.nn.Identity()
#         feature_extractor.heads = torch.nn.Identity()
#     elif name == "vivit":
#         image_processor = VivitImageProcessor.from_pretrained(
#             "google/vivit-b-16x2-kinetics400"
#         )
#         model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
#
#     # elif name == "swin":
#     #     weights = Swin3D_T_Weights.DEFAULT
#     #     feature_extractor = swin3d_t(weights=weights)
#     #     feature_extractor.heads = torch.nn.Identity()
#
#     # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#     feature_extractor = feature_extractor.to(device)
#     # Set the model to evaluation mode
#     feature_extractor = feature_extractor.eval()
#     return feature_extractor
#
#
# def extract_features(video_data_raw, feature_extractor, transforms_to_apply=None):
#     # Load the pre-trained ResNet-101 model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if method in ["omnivore", "slowfast", "x3d_pca_nc64", "3dresnet"]:
#         video_data_for_transform = {"video": video_data_raw, "audio": None}
#         # print(f"Before Transformation: {video_data_for_transform['video'].size()}")
#     if method not in ["mvit"]:
#         video_data = transforms_to_apply(video_data_for_transform)
#         if method in ["omnivore"]:
#             video_inputs = video_data["video"]
#             # print(f"Omnivore After Transformation: {video_inputs.size()}")
#             video_input = video_inputs[0][None, ...].to(device)
#         elif method == "slowfast":
#             video_inputs = video_data["video"]
#             # print(f"Slowfast After Transformation: {video_inputs.size()}")
#             video_input = [i.to(device)[None, ...] for i in video_inputs]
#         elif method == "x3d_pca_nc64":
#             video_inputs = video_data["video"]
#             # print(f"x3d After Transformation: {video_inputs.size()}")
#             video_input = video_inputs.unsqueeze(0).to(device)
#         elif method == "3dresnet":
#             video_inputs = video_data["video"]
#             video_input = video_inputs.unsqueeze(0).to(device)
#
#     else:
#         if method in ["mvit", "swin"]:
#             video_data_raw = video_data_raw.swapaxes(0, 1).to(device)
#             video_input = transforms_to_apply(video_data_raw).unsqueeze(0)
#             # video_input = video_data_raw.to(device).unsqueeze(0)
#     with torch.no_grad():
#         features = feature_extractor(video_input)
#
#     return features.cpu().numpy()
#
#
# parser = argparse.ArgumentParser(description="Script for processing methods.")
# parser.add_argument(
#     "--backbone", type=str, required=True, help="Specify the method to be used."
# )
#
# args = parser.parse_args()
# method = args.backbone
#
# video_files_path = "/data/error_detection/dataset/videos/recordings"
# output_features_path = f"/data/error_detection/dataset/features/{method}"
#
# num_frames_per_feature = 30
# fps = 30
# mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]
# this_time = datetime.datetime.now()
# import wandb
#
# wandb.init(project=f"get_features{method}")
#
#
# # Initialize empty lists to hold all the data
# def process_video(file):
#     segment_size = fps / num_frames_per_feature
#     all_features = []
#     video_path = os.path.join(video_files_path, file)
#     video_name = os.path.splitext(os.path.basename(file))[0]
#     output_file_directory = os.path.join(output_features_path, video_name)
#     os.makedirs(output_file_directory, exist_ok=True)
#     video = EncodedVideo.from_path(video_path)
#     video_duration_frac = video.duration
#     video_duration_str = str(video_duration_frac)
#     num, den = video_duration_str, "1"
#     if "/" in video_duration_str:
#         num, den = video_duration_str.split("/")
#     video_duration = float(num) / float(den)
#     print(f"video: {video_name} video_duration: {video_duration} s")
#     segment_end = max(video_duration - segment_size + 1, 1)
#     if npy_files := glob.glob(os.path.join(output_file_directory, "*.npy")):
#         last_npy_file = natsorted(npy_files)[-1]
#         last_end_time = float(
#             os.path.basename(last_npy_file).split("_")[-1].split(".")[0]
#         )
#         print(f"Skipping {last_end_time} frames for video {video_name}")
#     else:
#         last_end_time = 0
#
#     for time in np.arange(last_end_time, segment_end):
#         start_time = time
#         end_time = start_time + segment_size
#         end_time = min(end_time, video_duration)
#         filename = f"{video_name}_{start_time}_{end_time}"
#         output_file_path = os.path.join(output_file_directory, filename)
#         if os.path.exists(f"{output_file_path}.npy"):
#             print(f"Skipping: {output_file_path}")
#             continue
#         video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
#         video_inputs = video_data["video"]
#         try:
#             features = extract_features(
#                 video_inputs,
#                 feature_extractor=feature_extractor,
#                 transforms_to_apply=video_transform,
#             )
#             np.save(output_file_path, features)
#             tqdm.write(
#                 f"Processing: {file}, Features Shape: {features.shape}, {start_time} - {end_time}"
#             )
#         except Exception as e:
#             print(f"Error: {e}")
#             print(f"Skipping: {file}")
#             # Save the filename to error.txt
#             with open(f"{method}_error.txt", "a") as f:
#                 f.write(f"{this_time} : Error in - {output_file_path}\n")
#             continue
#
#
# video_transform = get_video_transformation(method)
# feature_extractor = get_feature_extractor(method)
# num_threads = 10
# with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
#     list(tqdm(executor.map(process_video, mp4_files), total=len(mp4_files)))
# # process_video(mp4_files[0])


# -----------------------------------------------------------------------------------------------------------


# import argparse
# import glob
# import av
# import datetime
# import time
# from loguru import logger
# import numpy as np
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Lambda
# from tqdm import tqdm
# import os
# from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample,
#     UniformCropVideo,
# )
# from PIL import Image
# from natsort import natsorted
# import torchvision.transforms as T
# import torch
# from omnivore_local.transforms import SpatialCrop, TemporalCrop
# from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
# from torchvision.models.video import (
#     mvit_v2_s,
#     MViT_V2_S_Weights,
# )
# from transformers import (
#     VivitImageProcessor,
#     VivitForVideoClassification,
#     VivitModel,
#     AutoImageProcessor,
#     VideoMAEForVideoClassification,
#     VideoMAEModel,
# )
# import numpy as np
# import av
# from tqdm import tqdm
#
# # def get_video_transformation(name):
# #     # Define the specific transformations
#
# #     return video_transform
#
#
# def get_feature_extractor(name, device="cuda"):
#     if name == "vivit":
#         image_processor = VivitImageProcessor.from_pretrained(
#             "google/vivit-b-16x2-kinetics400",
#             size={"shortest_edge": 256, "longest_edge": 256},
#         )
#         # model = VivitForVideoClassification.from_pretrained(
#         #     "google/vivit-b-16x2-kinetics400"
#         # )
#         model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").to(device)
#
#     elif name == "mvit":
#         # model_name = "mvit_base_32x3"
#         # feature_extractor = torch.hub.load(
#         #     "facebookresearch/pytorchvideo", model_name, pretrained=True
#         # )
#         weights = MViT_V2_S_Weights.DEFAULT
#         image_processor = weights
#         model = mvit_v2_s(weights=weights)
#         # feature_extractor.heads = torch.nn.Identity()
#         model.heads = torch.nn.Identity()
#         model.to(device)
#         model.eval()
#     elif name == "video_mae":
#         image_processor = AutoImageProcessor.from_pretrained(
#             "MCG-NJU/videomae-base-finetuned-kinetics"
#         )
#         model = VideoMAEForVideoClassification.from_pretrained(
#             "MCG-NJU/videomae-base-finetuned-kinetics"
#         ).to(device)
#         # model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
#
#     # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#     # Set the model to evaluation mode
#     return image_processor, model
#
#
# @torch.no_grad()
# def extract_features(video_data_raw, image_processor, model):
#     # Load the pre-trained ResNet-101 model
#     # video_data_raw = video_data_raw.to_ndarray(format="rgb24")
#     # video_data_raw = video_data_raw.unsqueeze(0)
#     # video_data_raw = video_data_raw.permute(0, 2, 1, 3, 4)
#     # video_data_raw = video_data_raw.permute(1, 2, 3, 0)
#     if method in ["vivit", "video_mae"]:
#         inputs = image_processor(video_data_raw, return_tensors="pt").to("cuda")
#     elif method == "mvit":
#         # Step 2: Initialize the inference transforms
#         weights = MViT_V2_S_Weights.DEFAULT
#         video_transform = weights.transforms()
#         video_data_raw = torch.from_numpy(video_data_raw).to("cuda")
#         inputs = video_data_raw.permute(0, 3, 1, 2)
#         inputs = video_transform(inputs).float().unsqueeze(0)
#
#         # inputs = video_transform(video_data_raw)
#
#     # forward pass
#     with torch.no_grad():
#         if method in ["vivit"]:
#             outputs = model(**inputs)
#             last_hidden_state = outputs.pooler_output
#         elif method == "video_mae":
#             outputs = model(**inputs)
#             last_hidden_state = outputs.logits
#         else:
#             outputs = model(inputs)
#             last_hidden_state = outputs
#     return last_hidden_state
#
#
# parser = argparse.ArgumentParser(description="Script for processing methods.")
#
# parser.add_argument(
#     "--backbone", type=str, required=True, help="Specify the method to be used."
# )
#
# args = parser.parse_args()
# method = args.backbone
#
# # method = "video_mae"
# # method = "vivit"
#
#
# # video_files_path = "/data/error_detection/dataset/videos/recordings"
# frame_files_path = "/data/error_detection/dataset/frames"
# output_features_path = f"/data/error_detection/dataset/features/{method}"
#
# # for markov
# # video_files_path = (
# #     "/bigdata/akshay/datacollection/error_detection/dataset/videos/recordings"
# # )
# # output_features_path = (
# #     f"/bigdata/akshay/datacollection/error_detection/dataset/features/{method}"
# # )
#
# if method == "vivit":
#     num_frames_per_feature = 32
# elif method == "video_mae":
#     num_frames_per_feature = 16
# fps = 30
# # mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]
#
# directories = [
#     d
#     for d in os.listdir(frame_files_path)
#     if os.path.isdir(os.path.join(frame_files_path, d))
# ]
#
# mp4_files = natsorted(directories)
# this_time = datetime.datetime.now()
# import wandb
#
# wandb.init(project=f"get_features{method}")
#
#
# @torch.no_grad()
# def process_frames_from_path(path, fps, batch_size=128):
#     path = os.path.join(frame_files_path, path)
#     video_name = os.path.basename(path)
#     output_file_directory = os.path.join(output_features_path, os.path.basename(path))
#     os.makedirs(output_file_directory, exist_ok=True)
#
#     frame_files = natsorted(
#         glob.glob(os.path.join(path, "*.jpg"))
#     )  # Assume frames are in jpg format
#     total_frames = len(frame_files)
#     # if npy_files := glob.glob(os.path.join(output_file_directory, "*.npy")):
#     #     last_npy_file = natsorted(npy_files)[-1]
#     #     last_end_time = float(
#     #         os.path.basename(last_npy_file).split("_")[-1].split(".")[0]
#     #     )
#     #     last_end_frame = max(0, int(last_end_time * fps))
#     #     logger.info(f"Skipping {last_end_frame} frames for video {video_name}")
#     # else:
#     last_end_frame = 0
#     feature_batches = []
#
#     # Read and process frames in batches
#     for i in range(last_end_frame, total_frames, fps):
#         start_time = i / fps
#         end_time = (i + fps) / fps
#         tmp_output_file_path = os.path.join(
#             output_file_directory,
#             f"{video_name}_{start_time}_{end_time}",
#         )
#         if os.path.exists(f"{tmp_output_file_path}.npy"):
#             # print(f"Skipping: {tmp_output_file_path}")
#             continue
#         try:
#             batch = [
#                 np.array(Image.open(frame_files[j]))
#                 for j in range(i, min(i + num_frames_per_feature, total_frames))
#             ]
#             feature_batches.append(batch)
#
#             if len(feature_batches) >= batch_size:
#                 features = (
#                     extract_features(
#                         feature_batches,
#                         image_processor=image_processor,
#                         model=model,
#                     )
#                     .cpu()
#                     .numpy()
#                 )
#
#                 for j in range(batch_size):
#                     # get the start time stamp of this batch and then add j*fps to get the start time stamp of this feature
#                     cur_start_time = (i - ((batch_size - 1) * fps) + j * fps) / fps
#                     cur_end_time = cur_start_time + 1
#                     output_file_path = os.path.join(
#                         output_file_directory,
#                         f"{video_name}_{cur_start_time}_{cur_end_time}",
#                     )
#                     np.save(output_file_path, features[j])
#                     # tqdm.write(
#                     #     f"Processing: {path}, Features Shape: {features[j].shape}, Start: {cur_start_time}, End: {cur_end_time}"
#                     # )
#
#                 feature_batches = []  # Reset for the next batch
#
#         except Exception as e:
#             print(f"Error: {e}")
#             print(f"Skipping: {path}")
#             with open(f"{method}_error.txt", "a") as f:
#                 date = datetime.datetime.now()
#                 f.write(f"{date} Error in - {path}\n")
#
#
# # num_threads = 20
# # with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
# #     list(tqdm(executor.map(process_video, mp4_files), total=len(mp4_files)))
# image_processor, model = get_feature_extractor(method)
# for mp4_file in tqdm(mp4_files):
#     process_frames_from_path(mp4_file, fps=30, batch_size=128)
