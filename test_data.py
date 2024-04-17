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


def main():
    recording_id = "10_16"
    # check_file_match(recording_id)
    video_feature_path = f"/data/rohith/captain_cook/features/gopro/segments/imagebind_2/{recording_id}_360p.mp4.npz"
    video_numpy_data = load_video_embeddings(video_feature_path)
    print(video_numpy_data.shape)


if __name__ == "__main__":
    main()
