import os
import sys
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tts.modules.encoder.encoder import Encoder
from tts.modules.synthesizer.synthesizer import Synthesizer
from tts.modules.vocoder.vocoder import Vocoder

class TTS:
    def __init__(self):
        device = torch.device("cpu")
        self.encoder = Encoder(device)
        self.encoder.prepare_for_inference()
        self.synthesizer = Synthesizer()
        self.synthesizer.load_model(os.path.join(os.path.dirname(__file__), "..\\tts\\models\\synthesizer.pt"))
        self.vocoder = Vocoder() 
        self.vocoder.load_model(os.path.join(os.path.dirname(__file__), "..\\tts\\models\\vocoder.pt"))
        self.input_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\input")
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
        self.output_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\output")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def infere(self, audio_filename, text):
        audio_path = self.input_folder + "\\" + audio_filename
        assert os.path.exists(audio_path) == True, "audio file doesn't exist, please ensure that it exists in \"demo\\input\" folder!!"
        p_bar = tqdm(range(4), desc="Generating English Audio (voice-cloned)", disable=False)

        embeddings = self.encoder.get_embeddings_from_audio(audio_path)
        p_bar.update(1)

        texts = [text]
        embeds = [embeddings]
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        p_bar.update(2)

        generated_wav = self.vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.SAMPLING_RATE), mode="constant")
        p_bar.update(3)

        num_generated = len(os.listdir(self.output_folder)) + 1
        output_filename = "\\generated_output_%02d.wav" % num_generated
        output_path = self.output_folder + output_filename
        sf.write(output_path, generated_wav.astype(np.float32), self.synthesizer.SAMPLING_RATE)
        p_bar.update(4)
        print("\n"+"="*60 + "\nSaved output in \"demo\\output\" as %s\n" % output_filename + "="*60)


if __name__ == "__main__":
    tts = TTS()
    tts.infere("./common_voice_es_24989771.mp3", "I love to spend the weekend with my family, or to go play some games with the friends. Especially playing baseball and volleyball.")