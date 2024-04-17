import os
import torch
from tqdm import tqdm

from lib.imagebind.imagebind import data
from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType





def fetch_embeddings(audio_path):
    audio_paths = ["/data/rohith/captain_cook/audios/resolution_360p/1_7.wav"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Load data
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }
    embeddings = None
    with torch.no_grad():
        embeddings = model(inputs)
    print("Processed audio embeddings")


    print(
        "Audio : ",
        torch.softmax(embeddings[ModalityType.AUDIO], dim=-1),
    )


def main():
    audio_directory_path = "/data/rohith/captain_cook/audios/resolution_360p/"
    audio_feature_directory_path = "/data/rohith/captain_cook/features/gopro/audios"
    audio_files = os.listdir(audio_directory_path)

    # Use tqdm to show progress bar
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(audio_directory_path, audio_file)
        audio_embeddings = fetch_embeddings(audio_path)

        # Store embeddings in a npz file
        npz_file_path = os.path.join(audio_feature_directory_path, audio_file + ".npz")
        torch.save(audio_embeddings, npz_file_path)

if __name__ == "__main__":
    main()


