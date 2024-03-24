from moviepy.editor import *
from multiprocessing import Pool
import os


def extract_audio(mp4_file):
    try:
        print("---------------------------------------------------")
        # Construct the output file path
        base_name = os.path.basename(mp4_file)
        file_name, _ = os.path.splitext(base_name)
        print(f"Extracting from file {file_name}")
        output_file = os.path.join(output_directory, f"{file_name[:-5]}.wav")

        # Load the video file
        video = VideoFileClip(mp4_file)
        # Extract the audio
        audio = video.audio
        # Write the audio (as WAV)
        audio.write_audiofile(output_file, codec='pcm_s16le')  # codec for WAV format
        # Close the audio and video files to release resources
        audio.close()
        video.close()
        print(f"Extracted audio to {output_file}")
    except Exception as e:
        print(f"Failed to process {mp4_file}: {e}")


def main():
    input_directory = '/data/rohith/captain_cook/videos/resolution_360p/'
    global output_directory
    output_directory = '/data/rohith/captain_cook/audios/resolution_360p/'
    os.makedirs(output_directory, exist_ok=True)

    input_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.endswith('.mp4')]

    # Use Pool() without arguments will use as many processes as your machine's CPU supports
    with Pool() as pool:
        pool.map(extract_audio, input_files)


if __name__ == '__main__':
    main()
