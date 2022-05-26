import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules.encoder.encoder import Encoder

class TTS:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(device)
        self.encoder.prepare_for_inference()
    
    def infere(self, audio_path):
        embeddings = self.encoder.get_embeddings_from_audio(audio_path)