import os
import sys
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from TTS.modules.encoder.encoder import Encoder
from TTS.modules.synthesizer.synthesizer import Synthesizer
from TTS.modules.vocoder.vocoder import Vocoder
from NMT.transformer import eng_custom_standardization, spa_custom_standardization, translate
from SpeechRecognizer.speech_recognizer import SpeechRecognizer
from LipSync.inference import LipSyncing


class Pipeline:
    def __init__(self):

        #################################################################### Speech Recognizer ####################################################################
        self.speech_recognizer = SpeechRecognizer()

        #################################################################### Voice Cloning #############################################################################
        device = torch.device("cpu")
        
        print('Before Loading Encoder' ,torch.cuda.memory_allocated('cuda'))

        self.encoder = Encoder(device)
        self.encoder.prepare_for_inference()

        print('Before Loading Synthesizer' ,torch.cuda.memory_allocated('cuda'))
        self.synthesizer = Synthesizer()
        self.synthesizer.load_model(os.path.join(os.path.dirname(__file__), "..\\TTS\\models\\synthesizer.pt"))

        print('Before Loading Vocoder' ,torch.cuda.memory_allocated('cuda'))
        self.vocoder = Vocoder() 
        self.vocoder.load_model(os.path.join(os.path.dirname(__file__), "..\\TTS\\models\\vocoder.pt"))
        self.input_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\input")
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
        self.output_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\output")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        
    def generate_spanish_to_english_speech(self,video_name):

        speech_recognition_progress = tqdm(range(1), desc="Speech Recognition", disable=False)
        spanish_text, audio_filename= self.speech_recognizer.get_text_of_audio(video_name)
        speech_recognition_progress.update(1)
        speech_recognition_progress.close()
        
        audio_path = self.input_folder + "\\" + audio_filename
        assert os.path.exists(audio_path) == True, "audio file doesn't exist, please ensure that it exists in \"demo\\input\" folder!!"
        
        print('Before Translating' ,torch.cuda.memory_allocated('cuda'))

        translation_progress = tqdm(range(1), desc="Translating to English", disable=False)
        english_text = translate(spanish_text, eng_custom_standardization=eng_custom_standardization, spa_custom_standardization=spa_custom_standardization)
        print(english_text)
        translation_progress.update(1)
        translation_progress.close()
        
        print('After Translating' ,torch.cuda.memory_allocated('cuda'))
        

        speech_generation_progress = tqdm(range(4), desc="Generating English Audio (voice-cloned)", disable=False)
        embeddings = self.encoder.get_embeddings_from_audio(audio_path)
        speech_generation_progress.update(1)

        texts = [english_text]
        embeds = [embeddings]
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        speech_generation_progress.update(1)

        generated_wav = self.vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.SAMPLING_RATE), mode="constant")
        speech_generation_progress.update(1)

        num_generated = len(os.listdir(self.output_folder)) + 1
        cloned_audio_filename = "\\generated_output_%02d.wav" % num_generated
        cloned_audio_path = self.output_folder + cloned_audio_filename
        sf.write(cloned_audio_path, generated_wav.astype(np.float32), self.synthesizer.SAMPLING_RATE)
        speech_generation_progress.update(1)
        speech_generation_progress.close()
        print("\n"+"="*60 + "\nSaved output in \"demo\\output\" as %s\n" % cloned_audio_filename + "="*60)
        
        print('Before Deleting voice cloning' ,torch.cuda.memory_allocated())

        self.vocoder.delete_model_from_memory()
        # torch.cpu.empty_cache()
        torch.cuda.empty_cache()
        print('After Deleting voice cloning' ,torch.cuda.memory_allocated())
        
        return self.input_folder, cloned_audio_path, video_name, self.output_folder

if __name__ == "__main__":
    print('Beginning Memory allocation' ,torch.cuda.memory_allocated('cuda'))
    pipeline = Pipeline()
    print('After Loading Memory allocation' ,torch.cuda.memory_allocated('cuda'))

    pipeline.generate_spanish_to_english_speech("test_4.mp4")
    # pipeline.generate_spanish_to_english_speech("./videotest_2.mp4", ["Me encanta jugar con mis amigas", "me encanta mi colegio"])