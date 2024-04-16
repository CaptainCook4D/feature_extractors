# Extract audio features of segments from audio files
from lib.imagebind.imagebind import data
import torch
from lib.imagebind.imagebind.models import imagebind_model
from lib.imagebind.imagebind.models.imagebind_model import ModalityType

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

