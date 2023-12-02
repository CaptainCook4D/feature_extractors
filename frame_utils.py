import logging

import cv2
import os
import concurrent.futures

from tqdm import tqdm

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
	os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
	def __init__(
			self,
			video_path,
			video_name,
			output_directory,
	):
		self.video_path = video_path
		self.video_name = video_name[:-4]
		
		self.output_directory = output_directory
		self.video_output_directory = os.path.join(self.output_directory, self.video_name)
		
		os.makedirs(self.video_output_directory, exist_ok=True)
		self.video = cv2.VideoCapture(self.video_path)
	
	def extract_frames(self, frame_rate=1):
		"""
        Extract frames from video and save them as images
        :param frame_rate: frame rate to extract frames
        :return: None
        """
		logger.info(f"[{self.video_name}] Began extracting frames from video")
		count = 0
		while self.video.isOpened():
			ret, frame = self.video.read()
			if ret:
				if count % frame_rate == 0:
					cv2.imwrite(os.path.join(self.video_output_directory, 'frame_{:06d}.jpg'.format(count)), frame)
				count += 1
			else:
				break
		self.video.release()
		logger.info(f"[{self.video_name}] Finished extracting frames from video")


class VideoDirectoryFrameExtractor:
	
	def __init__(
			self,
			input_video_directory_path,
			output_video_directory_path,
			max_workers=10
	):
		self.input_video_directory_path = input_video_directory_path
		self.output_video_directory_path = output_video_directory_path
		self.max_workers = max_workers
		
		os.makedirs(self.output_video_directory_path, exist_ok=True)
	
	def extract_frames_from_video(self, video_path, video_name):
		video_frame_extractor = VideoFrameExtractor(
			video_path=video_path,
			video_name=video_name,
			output_directory=self.output_video_directory_path
		)
		video_frame_extractor.extract_frames(frame_rate=1)
	
	def parallel_video_frame_extractor(self):
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			futures = []
			for video_name in os.listdir(self.input_video_directory_path):
				video_path = os.path.join(self.input_video_directory_path, video_name)
				futures.append(executor.submit(self.extract_frames_from_video, video_path, video_name))
			
			# Optional: if you want to handle the results or exceptions from each thread
			for future in concurrent.futures.as_completed(futures):
				try:
					future.result()  # This will re-raise any exception raised in the thread
				except Exception as e:
					print(f"An error occurred: {e}")


if __name__ == "__main__":
	video_directory_frame_extractor = VideoDirectoryFrameExtractor(
		"/data/rohith/captain_cook/videos/resolution_360p",
		"/data/rohith/captain_cook/frames/gopro/resolution_360p"
	)
	video_directory_frame_extractor.parallel_video_frame_extractor()
