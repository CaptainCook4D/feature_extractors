"""
We split the audio file into a fixed duration of 2 seconds - The last incomplete segment is discarded.
We use these clipped segments to extract audio embeddings using the imagebind model.

Store the audio embeddings into a npz file.
"""

import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from torchvision import transforms
from lib.imagebind.imagebind.data import waveform2melspec
from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType

# Load the model from checkpoint into the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def load_and_transform_audio_data(
        audio_path,
        sample_rate=16000,
        clip_duration=2.0,
        num_mel_bins=128,
        target_length=204,
        mean=-4.268,
        std=9.138
):
    audio_outputs = []
    waveform, sr = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    else:
        print("Sample rate is same as expected")

    # Calculate the number of samples in each clip
    clip_length_samples = sample_rate * clip_duration

    # Calculate the number of complete 2-second clips
    num_samples = waveform.size(1)
    total_clips = num_samples // clip_length_samples

    for i in range(int(total_clips)):
        start_sample = i * clip_length_samples
        end_sample = start_sample + clip_length_samples
        waveform_clip = waveform[:, int(start_sample):int(end_sample)]
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        audio_clip = [waveform_melspec]
        normalize = transforms.Normalize(mean=mean, std=std)
        audio_clip = [normalize(ac).to(device) for ac in audio_clip]
        audio_clip = torch.stack(audio_clip, dim=0)
        audio_outputs.append(audio_clip)

    transformed_audio = torch.stack(audio_outputs, dim=0)
    return transformed_audio


def fetch_audio_embeddings(audio_path):
    # Load and transform the data
    inputs = {
        ModalityType.AUDIO: load_and_transform_audio_data(audio_path),
    }
    with torch.no_grad():
        embeddings = model(inputs)
    print(f"Processed audio embeddings for {audio_path}")

    return embeddings[ModalityType.AUDIO]


def main():
    audio_directory_path = "/data/rohith/captain_cook/audios/resolution_360p/"
    audio_feature_directory_path = "/data/rohith/captain_cook/features/gopro/audios/imagebind/"
    os.makedirs(audio_feature_directory_path, exist_ok=True)
    audio_files = os.listdir(audio_directory_path)

    # Use tqdm to show progress bar
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(audio_directory_path, audio_file)
        npz_file_path = os.path.join(audio_feature_directory_path, audio_file + ".npz")

        if os.path.exists(npz_file_path):
            print(f"Skipping {audio_file}")
            continue

        # Fetch audio embeddings
        audio_embeddings = fetch_audio_embeddings(audio_path)
        numpy_video_embeddings = audio_embeddings.cpu().numpy()
        # Store embeddings in a npz file
        np.savez(npz_file_path, video_embeddings=numpy_video_embeddings)


if __name__ == "__main__":
    main()
