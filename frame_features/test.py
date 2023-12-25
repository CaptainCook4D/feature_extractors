import numpy as np

# Load the .npz file
npz_file = np.load('/data/rohith/captain_cook/features/gopro/tsm/10_50_360p/frame_000000.jpg.npz')

# List all files/arrays in the npz file
print("Contents of the NPZ file:")
for file in npz_file.files:
    print(file)
    print(npz_file[file])