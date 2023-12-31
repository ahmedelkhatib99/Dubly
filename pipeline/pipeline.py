import os
import getopt
import sys
import torch
import threading
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
from NMT2.translate import translate as translate_nmt2
from SMT.translate import SMT

class VideoPipeline:
    def __init__(self, mode):
        self.mode = mode
        self.lip_syncing = LipSyncing()
        self.input_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\input")
        self.output_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\output")
        #self.output_folder = os.path.join(os.path.dirname(__file__), "..\\..\\UI\\dubly-ui\\src\\output")
    
    def generate_video(self, video_name):
        if (self.mode == "verbose"): video_generation_progress = tqdm(range(1), desc="Generating Video", disable=False)
        output_files = os.listdir(self.output_folder)
        for i in range(len(output_files)-1, -1, -1):
            if ".wav" in output_files[i]:
                audio_file = self.output_folder + "\\" + output_files[i]
                break
        self.lip_syncing.generate_video(self.input_folder + '\\' + video_name, 
                                        audio_file, 
                                        self.output_folder + '\\videos\\' + video_name)
        if (self.mode == "verbose"):
            video_generation_progress.update(1)
            video_generation_progress.close()
            print("\n"+"="*60 + "\nSaved video output in \"demo\\output\\videos\" as %s\n" %  video_name + "="*60)
        else: 
            print(video_name)
            sys.stdout.flush()

class SpeechToSpeechPipeline:
    def __init__(self, mode, translation_model):
        self.mode = mode
        #################################################################### Speech Recognizer ####################################################################
        self.speech_recognizer = SpeechRecognizer(self.mode)
        
        #################################################################### Voice Cloning ####################################################################
        device = torch.device("cpu")
        self.encoder = Encoder(device)
        self.encoder.prepare_for_inference()

        self.synthesizer = Synthesizer()
        self.synthesizer.load_model(os.path.join(os.path.dirname(__file__), "..\\TTS\\models\\synthesizer.pt"))

        self.vocoder = Vocoder() 
        self.vocoder.load_model(os.path.join(os.path.dirname(__file__), "..\\TTS\\models\\vocoder.pt"))
        self.input_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\input")
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
        self.output_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\output")
        #self.output_folder = os.path.join(os.path.dirname(__file__), "..\\..\\UI\\dubly-ui\\src\\output")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder + "\\videos")
        
        if translation_model == 3: self.smt_model = SMT()

        
    def generate_spanish_to_english_speech(self, video_name, translation_model):

        if (self.mode == "verbose"): speech_recognition_progress = tqdm(range(1), desc="Speech Recognition", disable=False)
        spanish_text, audio_filename= self.speech_recognizer.get_text_of_audio(video_name)
        if (self.mode == "verbose"):
            speech_recognition_progress.update(1)
            speech_recognition_progress.close()
            print(spanish_text)
        
        audio_path = self.input_folder + "\\" + audio_filename
        assert os.path.exists(audio_path) == True, "audio file doesn't exist, please ensure that it exists in \"demo\\input\" folder!!"
        
        if (self.mode == "verbose"): translation_progress = tqdm(range(1), desc="Translating to English", disable=False)
        if(translation_model==1): 
            english_text = translate(spanish_text, eng_custom_standardization=eng_custom_standardization, spa_custom_standardization=spa_custom_standardization)
        elif(translation_model==2): 
            english_text = ' '.join([translate_nmt2(sentence) for sentence in spanish_text])
        elif(translation_model==3):
            english_text = ' '.join([self.smt_model.translate(sentence) for sentence in spanish_text])
        if (self.mode == "verbose"):
            translation_progress.update(1)
            translation_progress.close()
            print(english_text)

        if (self.mode == "verbose"): speech_generation_progress = tqdm(range(4), desc="Generating English Audio (voice-cloned)", disable=False)
        embeddings = self.encoder.get_embeddings_from_audio(audio_path)
        if (self.mode == "verbose"): speech_generation_progress.update(1)

        text = [english_text]
        embeds = [embeddings]
        specs = self.synthesizer.synthesize_spectrograms(text, embeds)
        spec = specs[0]
        if (self.mode == "verbose"): speech_generation_progress.update(1)

        generated_wav = self.vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.config.sampling_rate), mode="constant")
        if (self.mode == "verbose"): speech_generation_progress.update(1)
        self.vocoder.delete_model_from_memory()
        torch.cuda.empty_cache()

        num_generated = len(os.listdir(self.output_folder)) - 1
        self.cloned_audio_filename = "\\generated_output_%02d.wav" % num_generated
        cloned_audio_path = self.output_folder + self.cloned_audio_filename
        sf.write(cloned_audio_path, generated_wav.astype(np.float32), self.synthesizer.config.sampling_rate)
        if (self.mode == "verbose"):
            speech_generation_progress.update(1)
            speech_generation_progress.close()
            print("\n"+"="*60 + "\nSaved audio output in \"demo\\output\" as %s\n" % self.cloned_audio_filename + "="*60)
        
def execute_speech_pipline(video_name, mode, translation_model):
    speechPipeline = SpeechToSpeechPipeline(mode, translation_model)
    speechPipeline.generate_spanish_to_english_speech(video_name, translation_model)

def execute_video_pipeline(video_name, mode):
    videoPipeline = VideoPipeline(mode)
    videoPipeline.generate_video(video_name)

def print_help_message():
    print("Please run the command as follows: python pipeline.py -f filename -m silent/verbose\n -t 1 or 2 or 3 is optional to choose translation model")
    sys.exit(2)

def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "hf:m:t:")
        if (len(opts) == 0) or (opts[0][0] == "-h"):
            print_help_message()
        elif (len(opts) >= 2 and not (opts[0][0] == "-f" and opts[1][0] == "-m" and (opts[1][1] in ["silent", "verbose"]))) or (len(opts)==3 and opts[2][0] == "-t" and (opts[1][1] in ["1", "2", "3"])):
                print_help_message()
        else:
            video_name = opts[0][1]
            mode = opts[1][1]
            translation_model = int(opts[2][1]) if len(opts)==3 else 2
            sys.argv = [sys.argv[0]]
            speech_thread = threading.Thread(target=lambda: execute_speech_pipline(video_name, mode, translation_model))
            speech_thread.start()
            speech_thread.join()

            video_thread = threading.Thread(target=lambda: execute_video_pipeline(video_name, mode))
            video_thread.start()
            video_thread.join()
    except getopt.GetoptError:
        print_help_message()

if __name__ == "__main__":
    main(sys.argv[1:])
