import os

import numpy as np


def fetch_numpy_array(data, filename):
    numpy_data = np.frombuffer(data[f"{filename}/data/0"], dtype=np.float32).reshape((-1, 1024))
    return numpy_data


def check_file_match(recording_id):
    audio_feature_path = f"/data/rohith/captain_cook/features/gopro/audios/imagebind/{recording_id}.wav.npz"
    video_feature_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_2/{recording_id}_360p.mp4.npz"

    audio_data = np.load(audio_feature_path)
    video_data = np.load(video_feature_path)

    audio_numpy_data = fetch_numpy_array(audio_data, f"{recording_id}.wav")
    video_numpy_data = fetch_numpy_array(video_data, f"{recording_id}_360p.mp4")

    print(audio_numpy_data.shape)
    print(video_numpy_data.shape)

    assert audio_numpy_data.shape == video_numpy_data.shape
    return audio_numpy_data, video_numpy_data


def load_video_embeddings(video_feature_path):
    video_data = np.load(video_feature_path)
    video_numpy_data = video_data["video_embeddings"]
    return video_numpy_data


def test_npz():
    import numpy as np

    # Load the Depth npz files
    depth_npz_directory = '/data/rohith/captain_cook/features/gopro/segments_2/depth/'
    depth_npz_files = os.listdir(depth_npz_directory)

    # Load the Text npz files
    text_npz_directory = '/data/rohith/captain_cook/features/gopro/segments_2/text/'
    text_npz_files = os.listdir(text_npz_directory)

    # Load the Video npz files
    video_npz_directory = '/data/rohith/captain_cook/features/gopro/segments_2/video/'
    video_npz_files = os.listdir(video_npz_directory)

    # Load the Audio npz files
    audio_npz_directory = '/data/rohith/captain_cook/features/gopro/segments_2/audio/'
    audio_npz_files = os.listdir(audio_npz_directory)

    # Check if all have the same shape and number of files
    assert len(depth_npz_files) == len(text_npz_files) == len(video_npz_files) == len(audio_npz_files)

    print("All npz directories have the same number of files")

    mismatch_counter = 0

    for depth_npz_file in depth_npz_files:
        recording_id = depth_npz_file.split(".")[0]

        depth_npz_file_path = os.path.join(depth_npz_directory, depth_npz_file)
        with np.load(depth_npz_file_path) as depth_npz_file_data:
            depth_npz_data = depth_npz_file_data['video_embeddings']

        text_npz_file = f"{recording_id}_360p.npz"
        text_npz_file_path = os.path.join(text_npz_directory, text_npz_file)
        with np.load(text_npz_file_path) as text_npz_file_data:
            text_npz_data = text_npz_file_data['video_embeddings']

        video_npz_file = f"{recording_id}_360p.mp4.npz"
        video_npz_file_path = os.path.join(video_npz_directory, video_npz_file)
        with np.load(video_npz_file_path) as video_npz_file_data:
            video_npz_data = video_npz_file_data['video_embeddings']

        audio_npz_file = f"{recording_id}.wav.npz"
        audio_npz_file_path = os.path.join(audio_npz_directory, audio_npz_file)
        with np.load(audio_npz_file_path) as audio_npz_file_data:
            audio_npz_data = audio_npz_file_data['video_embeddings']

        # Shape Check
        # assert text_npz_data.shape == video_npz_data.shape == audio_npz_data.shape == depth_npz_data.shape
        if text_npz_data.shape == video_npz_data.shape == audio_npz_data.shape == depth_npz_data.shape:
            continue
        else:
            mismatch_counter += 1
            print("-----------------------------------------------------------")
            print(f"[{recording_id}][{mismatch_counter}] Text: {text_npz_data.shape}, Video: {video_npz_data.shape}, Audio: {audio_npz_data.shape}, Depth: {depth_npz_data.shape}")

            print("-----------------------------------------------------------")


def test_pkl():
    import os
    import pickle as pkl

    def is_pickle_empty(pickle_file_path):
        """Check if a pickle file is empty or not."""
        # Check if the file exists and is not empty
        if os.path.exists(pickle_file_path) and os.path.getsize(pickle_file_path) > 0:
            return False  # File exists and has content
        else:
            return True  # File does not exist or is empty

    # Replace with the path to your pickle file
    pickle_file_path = '/data/bhavya/splits/ce_wts.pkl'

    # Check if the pickle is empty
    empty = is_pickle_empty(pickle_file_path)
    print(f"The pickle file is {'empty' if empty else 'not empty'}.")


    # Load the pickle file
    with open(pickle_file_path, 'rb') as file:
        data = pkl.load(file)
        print(data)



def main():
    recording_id = "10_16"
    # check_file_match(recording_id)
    video_feature_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_2/{recording_id}_360p.mp4.npz"
    video_numpy_data = load_video_embeddings(video_feature_path)
    print(video_numpy_data.shape)


if __name__ == "__main__":
    test_npz()
