import os
import subprocess

video_frames_directories_path = "/data/rohith/captain_cook/frames/gopro/resolution_360p/"

output_features_path = f"/data/rohith/captain_cook/features/gopro/frames/tsm/"

completed_videos = [folder.split(".")[0] for folder in os.listdir(output_features_path)]

video_folders = [folder for folder in os.listdir(video_frames_directories_path) if folder not in completed_videos]

batch_1 = video_folders[:5]
batch_2 = video_folders[5:]

batches = [batch_1, batch_2]

program = "frame_features/generate_frame_features.py"

for batch in batches:
    batch = ','.join(batch)
    command = ["python3", program, "--batch", batch]
    subprocess.run(command, check=True)