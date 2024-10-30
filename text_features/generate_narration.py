# Git Clone LaViLa
# Install respective requirements
# Place this file in that repo to generate narration text for each second of the video
# This method samples 4 frames from each second to generate the narration text for the video

import os
import urllib.request
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord
from tqdm import tqdm
import numpy as np
import json

from eval_narrator import decode_one
from lavila.data.datasets import video_loader_by_frames
from lavila.data.video_transforms import Permute
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_BASE_GPT2
from lavila.models.tokenizer import MyGPT2Tokenizer


def load_model(device):
    ckpt_name = 'vclm_openai_timesformer_base_gpt2_base.pt_ego4d.jobid_319630.ep_0002.md5sum_68a71f.pth'
    ckpt_path = os.path.join('modelzoo/', ckpt_name)
    os.makedirs('modelzoo/', exist_ok=True)
    if not os.path.exists(ckpt_path):
        print('Downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name),
                                   ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print('Loading model from checkpoint')
    model = VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,
        freeze_visual_vclm=False,
        freeze_visual_vclm_temporal=False,
        num_frames=4,
        drop_path_rate=0.
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)

    # Transforms on input frames
    crop_size = 224
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001],
                                        std=[68.5005327, 66.6321579, 70.32316305])
    ])

    return model, tokenizer, val_transform


def generate_video_narration(video_path, output_file, model, tokenizer, val_transform, device):
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps  # in seconds

    narrations_dict = {}

    num_seconds = int(duration)
    for sec in tqdm(range(1, num_seconds + 1)):
        # Get frame indices for this second
        start_frame = int((sec - 1) * fps)
        end_frame = int(sec * fps)

        # Sample 4 frames within this second
        frame_indices = np.linspace(start_frame, min(end_frame - 1, num_frames - 1), num=4, dtype=int)

        # Read frames
        frames = video_loader_by_frames('./', video_path, frame_indices)

        # Transform frames
        frames = val_transform(frames)
        frames = frames.unsqueeze(0)  # Add batch dimension
        frames = frames.to(device)

        with torch.no_grad():
            image_features = model.encode_image(frames)
            generated_text_ids, ppls = model.generate(
                image_features,
                tokenizer,
                target=None,  # Free-form generation
                max_text_length=77,
                top_k=None,
                top_p=0.95,  # Nucleus sampling
                num_return_sequences=10,  # Number of candidates: 10
                temperature=0.7,
                early_stopping=True,
            )

        # Collect generated texts
        generated_texts = []
        for i in range(10):
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            generated_texts.append(generated_text_str)

        # Store in dictionary
        narrations_dict[str(sec)] = generated_texts

    # Save narrations_dict to JSON
    with open(output_file, 'w') as f:
        json.dump(narrations_dict, f, indent=2)


def extract_narrations(model, tokenizer, val_transform, device):
    os.makedirs(output_directory_path, exist_ok=True)

    video_list = os.listdir(videos_directory_path)
    filtered_list = [video for video in video_list if int(video[0]) == 9]
    # Shuffle the list to distribute the load
    np.random.shuffle(filtered_list)

    for video in tqdm(filtered_list):
        if video.endswith(".mp4"):
            print(f"Processing {video}")
            video_path = os.path.join(videos_directory_path, video)
            output_file = os.path.join(output_directory_path, f"{os.path.splitext(video)[0]}.json")

            if os.path.exists(output_file):
                print("------------------------------------------------")
                print(f"Narrations already extracted for {video}")
                continue

            generate_video_narration(video_path, output_file, model, tokenizer, val_transform, device)
            print(f"Generated narrations saved to {output_file}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, val_transform = load_model(device)
    videos_directory_path = "/data/rohith/captain_cook/videos/gopro/resolution_360p"
    output_directory_path = "/data/rohith/captain_cook/narrations"
    extract_narrations(model, tokenizer, val_transform, device)
