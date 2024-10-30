# Extract textual features from text files
import argparse
import os

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


def fetch_text_embeddings(narration_path):
    transformed_text = load_and_transform_text(narration_path)

    # Initialize a list to hold the outputs
    output_embeddings = []

    # Process the video data in chunks of 100 using tqdm for progress visualization
    with torch.no_grad():
        for i in tqdm(range(0, len(transformed_text), 100), desc="Processing video chunks"):
            chunk = transformed_text[i:i + 100]  # Get the current chunk
            inputs = {
                ModalityType.TEXT: chunk
            }
            embeddings = model(inputs)
            output_embeddings.append(embeddings[ModalityType.VISION])

    # Stack all the collected embeddings into a single tensor
    stacked_embeddings = torch.cat(output_embeddings, dim=0)
    print(f"Processed segment text embeddings for {narration_path}")
    return stacked_embeddings


def main():
    narration_directory_path = "/data/rohith/captain_cook/narrations/"
    narration_feature_directory_path = f"/data/rohith/captain_cook/features/gopro/segments/text/"
    os.makedirs(narration_feature_directory_path, exist_ok=True)
    narration_files = os.listdir(narration_directory_path)

    # Use tqdm to show progress bar
    for narration_file in tqdm(narration_files):
        video_path = os.path.join(narration_directory_path, narration_file)
        npz_file_path = os.path.join(narration_feature_directory_path, narration_file + ".npz")

        if os.path.exists(npz_file_path):
            print(f"Skipping {narration_file}")
            continue

        # Fetch audio embeddings
        text_embeddings = fetch_text_embeddings(video_path)
        numpy_text_embeddings = text_embeddings.cpu().numpy()
        # Store embeddings in a npz file
        np.savez(npz_file_path, video_embeddings=numpy_text_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_length", type=float, required=True)

    segment_length = parser.parse_args().segment_length
    main()
