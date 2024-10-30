import argparse
import os
import json

import numpy as np
import torch
from tqdm import tqdm

from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType

from lib.imagebind.imagebind.data import BPE_PATH

from lib.imagebind.imagebind.models.multimodal_preprocessors import SimpleTokenizer

# Load the model from checkpoint into the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def load_and_transform_text(text):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def get_embeddings_for_lines(lines, batch_size=100):
    all_embeddings = []
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        transformed_text = load_and_transform_text(batch_lines)
        with torch.no_grad():
            inputs = {ModalityType.TEXT: transformed_text}
            embeddings = model(inputs)
            text_embeddings = embeddings[ModalityType.TEXT]
        all_embeddings.append(text_embeddings)
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def fetch_text_embeddings(narration_path):
    # Read the JSON file
    with open(narration_path, 'r') as f:
        narration_data = json.load(f)

    # Initialize a list to hold the averaged embeddings per second
    averaged_embeddings_per_second = []

    # Sort the seconds in numerical order
    seconds = sorted(narration_data.keys(), key=lambda x: int(x))

    for second in tqdm(seconds, desc="Processing seconds"):
        lines = narration_data[second]
        # Preprocess each line by replacing "#C C" with "actor"
        processed_lines = [line.replace("#C C", "actor") for line in lines]

        if not processed_lines:
            continue

        # Get embeddings for each line
        embeddings = get_embeddings_for_lines(processed_lines)

        # Average the embeddings for this second
        averaged_embedding = embeddings.mean(dim=0)

        averaged_embeddings_per_second.append(averaged_embedding)

    # Stack all the averaged embeddings into a single tensor
    if averaged_embeddings_per_second:
        stacked_embeddings = torch.stack(averaged_embeddings_per_second)
        print(f"Processed segment text embeddings for {narration_path}")
        return stacked_embeddings
    else:
        print(f"No embeddings were processed for {narration_path}")
        return None


def main():
    narration_directory_path = "/data/rohith/captain_cook/narrations/"
    narration_feature_directory_path = f"/data/rohith/captain_cook/features/gopro/segments/text/"
    os.makedirs(narration_feature_directory_path, exist_ok=True)
    narration_files = [f for f in os.listdir(narration_directory_path) if f.endswith('.json')]

    # Use tqdm to show progress bar
    for narration_file in tqdm(narration_files, desc="Processing files"):
        video_path = os.path.join(narration_directory_path, narration_file)
        npz_file_name = os.path.splitext(narration_file)[0] + '.npz'
        npz_file_path = os.path.join(narration_feature_directory_path, npz_file_name)

        if os.path.exists(npz_file_path):
            print(f"Skipping {narration_file}")
            continue

        # Fetch text embeddings
        text_embeddings = fetch_text_embeddings(video_path)
        if text_embeddings is not None:
            numpy_text_embeddings = text_embeddings.cpu().numpy()
            # Store embeddings in a npz file
            np.savez(npz_file_path, video_embeddings=numpy_text_embeddings)
        else:
            print(f"No embeddings to save for {narration_file}")


if __name__ == "__main__":
    main()
