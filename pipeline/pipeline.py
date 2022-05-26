import os
import sys
import torch
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules.encoder.encoder import Encoder
from modules.synthesizer.synthesizer import Synthesizer
from modules.vocoder.vocoder import Vocoder

class TTS:
    def __init__(self):
        device = torch.device("cpu")
        self.encoder = Encoder(device)
        self.encoder.prepare_for_inference()
        self.synthesizer = Synthesizer()
        self.synthesizer.load_model(Path("../models/synthesizer.pt"))
        self.vocoder = Vocoder() 
        self.vocoder.load_model(Path("../models//vocoder.pt"))
    
    def infere(self, audio_path, text):
        embeddings = self.encoder.get_embeddings_from_audio(audio_path)

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embeddings]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")

        ## Generating the waveform
        print("Synthesizing the waveform:")

        generated_wav = self.vocoder.infer_waveform(spec)


        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.SAMPLING_RATE), mode="constant")

        # # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        # generated_wav = self.encoder.get_embeddings_from_audio(generated_wav)

        # Play the audio (non-blocking)
        
        try:
            sd.stop()
            sd.play(generated_wav, self.synthesizer.SAMPLING_RATE)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
            print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
        except:
            raise

        # Save it on the disk
        num_generated = 1
        filename = "demo_output_%02d.wav" % num_generated
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), self.synthesizer.SAMPLING_RATE)
        print("\nSaved output as %s\n\n" % filename)


if __name__ == "__main__":
    tts = TTS()
    tts.infere("./common_voice_es_24989771.mp3", "I love my family a lot. Let's go play some games with the friends. I love playing baseball and volleyball.")