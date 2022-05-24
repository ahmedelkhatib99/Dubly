import os
import yaml
import librosa
import struct
import webrtcvad
import numpy as np
from scipy.ndimage.morphology import binary_dilation


class Encoder:
    def __init__(self):
        self.dataset_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\dataset")
        self.preprocessing_output_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\output\\preprocessing")
        with open(os.path.join(os.path.dirname(__file__), "..\\..\\configurations.yaml"), "r") as f:
            config = yaml.safe_load(f)
        self.global_sampling_rate = config["PREPROCESSING"]["SAMPLING_RATE"]
        self.average_amplitude_target_dBFS = config["PREPROCESSING"]["AMPLITUDE_DBFS"]
        voice_activity_window_msecs = config["PREPROCESSING"]["VAD_WINDOW_MS"]
        self.samples_per_voice_activity_window = (
            voice_activity_window_msecs * self.global_sampling_rate) // 1000
        self.moving_average_width = config["PREPROCESSING"]["VAD_MOVING_AVG_WIDTH"]
        self.mel_window_width = config["PREPROCESSING"]["MEL_WINDOW_WIDTH"]
        self.mel_window_step = config["PREPROCESSING"]["MEL_WINDOW_STEP"]
        self.mel_channels_count = config["PREPROCESSING"]["MEL_CHANNELS_COUNT"]
        self.frames_min_count = config["PREPROCESSING"]["FRAMES_MIN_COUNT"]

    def preprocess_dataset(self):
        speakers_dirs = [self.dataset_path + "\\" +
                         subdir for subdir in os.listdir(self.dataset_path)]
        speakers_with_audios = [{speaker_dir: self.get_speaker_audios(
            speaker_dir)} for speaker_dir in speakers_dirs]

        for speaker in speakers_with_audios:
            for speaker_path, audios_paths in speaker.items():
                for audio_path in audios_paths:
                    audio = self.get_audio_and_sampling_rate(audio_path)
                    audio = self.normalize_amplitude(audio)
                    audio = self.trim_extra(audio)
                    pcm = self.get_16_bit_pulse_code_modulation(audio)
                    voice_activity_detection = webrtcvad.Vad(mode=3)
                    is_window_contains_speech = self.detect_windows_containing_speech(
                        audio, pcm, voice_activity_detection)
                    is_window_contains_speech = self.fill_gaps(
                        is_window_contains_speech)
                    sample_positions = self.get_sample_positions(
                        is_window_contains_speech)
                    audio_samples = audio[sample_positions]
                    mel_frames = self.get_mel_frames(audio_samples)
                    if len(mel_frames) < self.frames_min_count:
                        continue
                    else:
                        self.save_mel(mel_frames, audio_path)

    def save_mel(self, mel_frames, audio_path):
        np.save(self.preprocessing_output_path + "\\" +
                os.path.split(audio_path)[1].split(".")[0]+'.npy', mel_frames)

    def get_mel_frames(self, audio_samples):
        mel_frames = librosa.feature.melspectrogram(
            audio_samples,
            self.global_sampling_rate,
            n_fft=int(self.global_sampling_rate *
                      self.mel_window_width / 1000),
            hop_length=int(self.global_sampling_rate *
                           self.mel_window_step / 1000),
            n_mels=self.mel_channels_count
        )
        mel_frames = mel_frames.astype(np.float32).T
        return mel_frames

    def get_sample_positions(self, is_window_contains_speech):
        return np.repeat(is_window_contains_speech, self.samples_per_voice_activity_window)

    def fill_gaps(self, is_window_contains_speech):
        structure_element = np.ones(self.samples_per_voice_activity_window + 1)
        return binary_dilation(is_window_contains_speech, structure_element)

    @staticmethod
    def smooth_windows_containing_speech_detection(is_window_contains_speech, moving_average_width):
        padded_flag_list = np.concatenate((np.zeros(
            (moving_average_width - 1) // 2), is_window_contains_speech, np.zeros(moving_average_width // 2)))
        smoothed_flag_list = np.cumsum(padded_flag_list, dtype=float)
        smoothed_flag_list[moving_average_width:] = smoothed_flag_list[moving_average_width:] - \
            smoothed_flag_list[:-moving_average_width]
        return smoothed_flag_list[moving_average_width - 1:] / moving_average_width

    def detect_windows_containing_speech(self, audio, pcm, voice_activity_detection):
        is_contains_speech = []
        for start in range(0, len(audio), self.samples_per_voice_activity_window):
            end = start + self.samples_per_voice_activity_window
            is_contains_speech.append(voice_activity_detection.is_speech(
                pcm[start * 2: end * 2], sample_rate=self.global_sampling_rate))
        is_contains_speech = np.array(is_contains_speech)
        is_contains_speech = self.smooth_windows_containing_speech_detection(
            is_contains_speech, self.moving_average_width)
        is_contains_speech = np.round(is_contains_speech).astype(np.bool)
        return is_contains_speech

    @staticmethod
    def get_16_bit_pulse_code_modulation(audio):
        return struct.pack("%dh" % len(audio), *(np.round(audio * ((2 ** 15) - 1))).astype(np.int16))

    def trim_extra(self, audio):
        return audio[:len(audio) - (len(audio) % self.samples_per_voice_activity_window)]

    def normalize_amplitude(self, audio):
        amplitude_error_dbFS = self.average_amplitude_target_dBFS - \
            10 * np.log10(np.mean(audio ** 2))
        if amplitude_error_dbFS > 0:
            audio = audio * (10 ** (amplitude_error_dbFS / 20))
        return audio

    def get_audio_and_sampling_rate(self, audio_path):
        audio, sampling_rate = librosa.load(audio_path, sr=None)
        if sampling_rate is not None and sampling_rate != self.global_sampling_rate:
            audio = librosa.resample(
                audio, sampling_rate, self.global_sampling_rate)
        return audio

    @staticmethod
    def get_speaker_audios(speaker_dir):
        return [speaker_dir + "\\" + file for file in os.listdir(speaker_dir) if os.path.isfile(speaker_dir + "\\" + file) and file.endswith(".mp3")]


encoder = Encoder()
encoder.preprocess_dataset()
