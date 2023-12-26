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
pickle_file_path = '/data/rohith/captain_cook/features/gopro/frames/tsm/10_48_360p/checkpoint.pkl'

# Check if the pickle is empty
empty = is_pickle_empty(pickle_file_path)
print(f"The pickle file is {'empty' if empty else 'not empty'}.")
