import os, sys
import librosa
import webrtcvad
import struct
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
        padded_flag_list = np.concatenate((np.zeros(
            (configs.moving_average_width - 1) // 2), is_window_contains_speech, np.zeros(configs.moving_average_width // 2)))
        smoothed_flag_list = np.cumsum(padded_flag_list, dtype=float)
        smoothed_flag_list[configs.moving_average_width:] = smoothed_flag_list[configs.moving_average_width:] - \
            smoothed_flag_list[:-configs.moving_average_width]
        return smoothed_flag_list[configs.moving_average_width - 1:] / configs.moving_average_width

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
            configs.samples_per_voice_activity_window + 1)
        return binary_dilation(is_window_contains_speech, structure_element)

    @staticmethod
    def save_mel(configs: EncoderConfiguration, mel_frames, audio_path):
        if not os.path.exists(configs.preprocessing_output_folder):
            os.makedirs(configs.preprocessing_output_folder)
        np.save(configs.preprocessing_output_folder + "\\" +
                os.path.split(audio_path)[1].split(".")[0]+'.npy', mel_frames)

    @staticmethod
    def get_melspectrograms_for_training_iteration(configs: EncoderConfiguration, current_training_iteration, loaded_mels, device):
        mels_start = (current_training_iteration * configs.mels_count_per_iteration) % len(loaded_mels)
        mels_end = len(loaded_mels) if mels_start + configs.mels_count_per_iteration > len(loaded_mels) else mels_start + configs.mels_count_per_iteration
        training_mels = loaded_mels[mels_start: mels_end]
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

        mels_count, samples_per_mel = embeddings.shape[:2]

        mels_centroids = mean(embeddings, dim=1, keepdim=True)
        mels_centroids = mels_centroids.clone(
        ) / (norm(mels_centroids, dim=2, keepdim=True) + 1e-5)

        samples_centroids = (
            sum(embeddings, dim=1, keepdim=True) - embeddings) / (samples_per_mel - 1)
        samples_centroids = samples_centroids.clone(
        ) / (norm(samples_centroids, dim=2, keepdim=True) + 1e-5)

        similarity_matrix = zeros(
            mels_count, samples_per_mel, mels_count).to(device)
        all_mels_positions = 1 - np.eye(mels_count, dtype=np.int)
        for mel_index in range(mels_count):
            mel_positions = np.where(all_mels_positions[mel_index])[0]
            similarity_matrix[mel_positions, :, mel_index] = (
                embeddings[mel_positions] * mels_centroids[mel_index]).sum(dim=2)
            similarity_matrix[mel_index, :, mel_index] = (
                embeddings[mel_index] * samples_centroids[mel_index]).sum(dim=1)

        return similarity_matrix * similarity_weight + similarity_bias

    @staticmethod
    def reshape_embeddings(configs: EncoderConfiguration, device, embeddings: Tensor):
        return embeddings.view((embeddings.shape[0], configs.global_frames_count, INFERRED_DIMENSION)).to(device)

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
