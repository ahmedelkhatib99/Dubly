import os, sys
import librosa
import webrtcvad
import struct
import random
import numpy as np
from torch import Tensor, no_grad, from_numpy, mean, norm, sum, zeros
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from encoder_config import EncoderConfiguration

INFERRED_DIMENSION = -1


class EncoderUtils:
    @staticmethod
    def get_speaker_audios(speaker_dir):
        return [speaker_dir + "\\" + file for file in os.listdir(speaker_dir) if os.path.isfile(speaker_dir + "\\" + file) and file.endswith(".mp3")]

    @staticmethod
    def convert_audio_to_mel_spectrogram_frames(configs: EncoderConfiguration, should_suppress_noise, audio):
        audio_samples = audio
        if should_suppress_noise:
            pcm = EncoderUtils.get_16_bit_pulse_code_modulation(audio)
            voice_activity_detection = webrtcvad.Vad(mode=3)
            is_window_contains_speech = EncoderUtils.detect_windows_containing_speech(configs, audio, pcm, voice_activity_detection)
            is_window_contains_speech = EncoderUtils.fill_gaps(configs, is_window_contains_speech)
            sample_positions = EncoderUtils.get_sample_positions(configs, is_window_contains_speech)
            audio_samples = audio[sample_positions]
        mel_frames = EncoderUtils.get_mel_frames(configs, audio_samples)
            
        return mel_frames

    @staticmethod
    def get_audio(configs: EncoderConfiguration, audio_path):
        audio = EncoderUtils.get_audio_with_correct_sampling_rate(configs, audio_path)
        audio = EncoderUtils.normalize_amplitude(configs, audio)
        audio = EncoderUtils.trim_extra(configs, audio)
        return audio

    @staticmethod
    def get_audio_with_correct_sampling_rate(configs: EncoderConfiguration, audio_path):
        audio, sampling_rate = librosa.load(audio_path, sr=None)
        if sampling_rate is not None and sampling_rate != configs.global_sampling_rate:
            audio = librosa.resample(
                audio, sampling_rate, configs.global_sampling_rate)
        return audio

    @staticmethod
    def normalize_amplitude(configs: EncoderConfiguration, audio):
        amplitude_error_dbFS = configs.average_amplitude_target_dBFS - \
            10 * np.log10(np.mean(audio ** 2))
        if amplitude_error_dbFS > 0:
            audio = audio * (10 ** (amplitude_error_dbFS / 20))
        return audio

    @staticmethod
    def trim_extra(configs: EncoderConfiguration, audio):
        return audio[:len(audio) - (len(audio) % configs.samples_per_voice_activity_window)]

    @staticmethod
    def get_16_bit_pulse_code_modulation(audio):
        return struct.pack("%dh" % len(audio), *(np.round(audio * ((2 ** 15) - 1))).astype(np.int16))

    @staticmethod
    def detect_windows_containing_speech(configs: EncoderConfiguration, audio, pcm, voice_activity_detection):
        is_contains_speech = []
        for start in range(0, len(audio), configs.samples_per_voice_activity_window):
            end = start + configs.samples_per_voice_activity_window
            is_contains_speech.append(voice_activity_detection.is_speech(pcm[start * 2: end * 2],
                                                                         sample_rate=configs.global_sampling_rate))
        is_contains_speech = np.array(is_contains_speech)
        is_contains_speech = EncoderUtils.smooth_windows_containing_speech_detection(configs, is_contains_speech)
        is_contains_speech = np.round(is_contains_speech).astype(np.bool)
        return is_contains_speech

    @staticmethod
    def smooth_windows_containing_speech_detection(configs: EncoderConfiguration, is_window_contains_speech):
        padded_is_window_contains_speech = np.concatenate((np.zeros(
            (configs.moving_average_width - 1) // 2), is_window_contains_speech, np.zeros(configs.moving_average_width // 2)))
        smoothed_is_window_contains_speech = np.cumsum(padded_is_window_contains_speech, dtype=float)
        smoothed_is_window_contains_speech[configs.moving_average_width:] = smoothed_is_window_contains_speech[configs.moving_average_width:] - \
            smoothed_is_window_contains_speech[:-configs.moving_average_width]
        return smoothed_is_window_contains_speech[configs.moving_average_width - 1:] / configs.moving_average_width

    @staticmethod
    def get_sample_positions(configs: EncoderConfiguration, is_window_contains_speech):
        return np.repeat(is_window_contains_speech, configs.samples_per_voice_activity_window)

    @staticmethod
    def get_mel_frames(configs: EncoderConfiguration, audio_samples):
        mel_frames = librosa.feature.melspectrogram(
            audio_samples,
            configs.global_sampling_rate,
            n_fft=int(configs.global_sampling_rate *
                      configs.mel_window_width / 1000),
            hop_length=int(configs.global_sampling_rate *
                           configs.mel_window_step / 1000),
            n_mels=configs.mel_channels_count
        )
        mel_frames = mel_frames.astype(np.float32).T
        return mel_frames

    @staticmethod
    def fill_gaps(configs: EncoderConfiguration, is_window_contains_speech):
        structure_element = np.ones(
            configs.max_silence + 1)
        return binary_dilation(is_window_contains_speech, structure_element)

    @staticmethod
    def save_mel(configs: EncoderConfiguration, mel_frames, audio_path):
        output_folder = configs.preprocessing_output_folder + "\\" + os.path.dirname(audio_path).split("\\")[-1]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        np.save(output_folder + "\\" + os.path.split(audio_path)[1].split(".")[0]+'.npy', mel_frames)

    @staticmethod
    def get_melspectrograms_for_training_iteration(configs: EncoderConfiguration, current_training_iteration, device):
        speakers = os.listdir(configs.preprocessing_output_folder)
        speakers_start = (current_training_iteration * configs.speakers_count_per_iteration) % len(speakers)
        speakers_end = len(speakers) if speakers_start + configs.speakers_count_per_iteration > len(speakers) else speakers_start + configs.speakers_count_per_iteration
        training_speakers = speakers[speakers_start: speakers_end]
        training_mels = []
        for speaker in tqdm(training_speakers, desc="Loading mel spectrograms"):
            speaker_mels = os.listdir(configs.preprocessing_output_folder + "\\" + speaker)
            for _ in range(configs.mels_count_per_speaker):
                choice = configs.preprocessing_output_folder + "\\" + speaker + "\\" + random.choice(speaker_mels)
                print("for " + speaker + " we choose: " + choice)
                training_mels.append(np.load(choice))
        training_frames = EncoderUtils.extract_frames_from_training_mels(configs, training_mels)
        return from_numpy(np.array(training_frames)).to(device)

    @staticmethod
    def extract_frames_from_training_mels(configs: EncoderConfiguration, training_mels):
        frames = []
        for mel in tqdm(training_mels, desc="Extracting training frames"):
            if mel.shape[0] == configs.global_frames_count:
                mel_sample_start = 0
            else:
                mel_sample_start = np.random.randint(
                    0, mel.shape[0] - configs.global_frames_count)
            mel_sample_end = mel_sample_start + configs.global_frames_count
            frames.append(mel[mel_sample_start: mel_sample_end])
        return frames

    @staticmethod
    def calculate_similarity_matrix(similarity_weight, similarity_bias, device, embeddings):

        speakers_count, mels_count_per_speaker = embeddings.shape[:2]

        speakers_centroids = mean(embeddings, dim=1, keepdim=True)
        speakers_centroids = speakers_centroids.clone() / (norm(speakers_centroids, dim=2, keepdim=True) + 1e-5)

        centroids_excluding_speaker_mels = (sum(embeddings, dim=1, keepdim=True) - embeddings) / (mels_count_per_speaker - 1)
        centroids_excluding_speaker_mels = centroids_excluding_speaker_mels.clone() / (norm(centroids_excluding_speaker_mels, dim=2, keepdim=True) + 1e-5)

        similarity_matrix = zeros(speakers_count, mels_count_per_speaker, speakers_count).to(device)
        speakers_positions = 1 - np.eye(speakers_count, dtype=np.int)
        for speaker_index in range(speakers_count):
            speaker_mels_positions = np.where(speakers_positions[speaker_index])[0]
            similarity_matrix[speaker_mels_positions, :, speaker_index] = (embeddings[speaker_mels_positions] * speakers_centroids[speaker_index]).sum(dim=2)
            similarity_matrix[speaker_index, :, speaker_index] = (embeddings[speaker_index] * centroids_excluding_speaker_mels[speaker_index]).sum(dim=1)

        return similarity_matrix * similarity_weight + similarity_bias

    @staticmethod
    def reshape_embeddings(configs: EncoderConfiguration, device, embeddings: Tensor):
        return embeddings.view((configs.speakers_count_per_iteration, configs.mels_count_per_speaker, INFERRED_DIMENSION)).to(device)

    @staticmethod
    def calculate_equal_error_rate(mels_count, sim_matrix, ground_truth):
        with no_grad():
            def inv_argmax(i): return np.eye(1, mels_count, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            false_positive_rates, true_positive_rate, _ = roc_curve(
                labels.flatten(), preds.flatten())
            equal_error_rate = brentq(
                lambda x: 1. - x - interp1d(false_positive_rates, true_positive_rate)(x), 0.0, 1.0)
        return equal_error_rate
    
    @staticmethod
    def get_partitioning_positions(configs: EncoderConfiguration, audio_length):
        samples_per_frame = int((configs.global_sampling_rate * configs.mel_window_step / 1000))
        frames_count = int(np.ceil((audio_length + 1) / samples_per_frame))
        step_size = max(int(np.round(configs.global_frames_count * 0.5)), 1)
        steps_count = max(1, frames_count - configs.global_frames_count + step_size + 1)
        audio_partition_positions = []
        mel_partition_positions = []
        for i in range(0, steps_count, step_size):
            mel_partition = np.array([i, i + configs.global_frames_count])
            audio_partition = mel_partition * samples_per_frame
            mel_partition_positions.append(slice(*mel_partition))
            audio_partition_positions.append(slice(*audio_partition))
        last_audio_parition_position = audio_partition_positions[-1]
        coverage = (audio_length - last_audio_parition_position.start) / (last_audio_parition_position.stop - last_audio_parition_position.start)
        if coverage < 0.75 and len(mel_partition_positions) > 1:
            mel_partition_positions = mel_partition_positions[:-1]
            audio_partition_positions = audio_partition_positions[:-1]
        return audio_partition_positions, mel_partition_positions
