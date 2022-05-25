import os
import librosa
import struct
import numpy as np
from torch import device, Tensor, no_grad, from_numpy, mean, norm, sum, zeros
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

INFERRED_DIMENSION = -1


class EncoderUtils:
    @staticmethod
    def get_speaker_audios(speaker_dir):
        return [speaker_dir + "\\" + file for file in os.listdir(speaker_dir) if os.path.isfile(speaker_dir + "\\" + file) and file.endswith(".mp3")]

    @staticmethod
    def get_audio_with_correct_sampling_rate(global_sampling_rate, audio_path):
        audio, sampling_rate = librosa.load(audio_path, sr=None)
        if sampling_rate is not None and sampling_rate != global_sampling_rate:
            audio = librosa.resample(
                audio, sampling_rate, global_sampling_rate)
        return audio

    @staticmethod
    def normalize_amplitude(average_amplitude_target_dBFS, audio):
        amplitude_error_dbFS = average_amplitude_target_dBFS - \
            10 * np.log10(np.mean(audio ** 2))
        if amplitude_error_dbFS > 0:
            audio = audio * (10 ** (amplitude_error_dbFS / 20))
        return audio

    @staticmethod
    def trim_extra(samples_per_voice_activity_window, audio):
        return audio[:len(audio) - (len(audio) % samples_per_voice_activity_window)]

    @staticmethod
    def get_16_bit_pulse_code_modulation(audio):
        return struct.pack("%dh" % len(audio), *(np.round(audio * ((2 ** 15) - 1))).astype(np.int16))

    @staticmethod
    def detect_windows_containing_speech(samples_per_voice_activity_window, global_sampling_rate, moving_average_width, audio, pcm, voice_activity_detection):
        is_contains_speech = []
        for start in range(0, len(audio), samples_per_voice_activity_window):
            end = start + samples_per_voice_activity_window
            is_contains_speech.append(voice_activity_detection.is_speech(pcm[start * 2: end * 2],
                                                                         sample_rate=global_sampling_rate))
        is_contains_speech = np.array(is_contains_speech)
        is_contains_speech = EncoderUtils.smooth_windows_containing_speech_detection(is_contains_speech,
                                                                                     moving_average_width)
        is_contains_speech = np.round(is_contains_speech).astype(np.bool)
        return is_contains_speech

    @staticmethod
    def smooth_windows_containing_speech_detection(is_window_contains_speech, moving_average_width):
        padded_flag_list = np.concatenate((np.zeros(
            (moving_average_width - 1) // 2), is_window_contains_speech, np.zeros(moving_average_width // 2)))
        smoothed_flag_list = np.cumsum(padded_flag_list, dtype=float)
        smoothed_flag_list[moving_average_width:] = smoothed_flag_list[moving_average_width:] - \
            smoothed_flag_list[:-moving_average_width]
        return smoothed_flag_list[moving_average_width - 1:] / moving_average_width

    @staticmethod
    def get_sample_positions(samples_per_voice_activity_window, is_window_contains_speech):
        return np.repeat(is_window_contains_speech, samples_per_voice_activity_window)

    @staticmethod
    def get_mel_frames(global_sampling_rate, mel_window_width, mel_window_step, mel_channels_count, audio_samples):
        mel_frames = librosa.feature.melspectrogram(
            audio_samples,
            global_sampling_rate,
            n_fft=int(global_sampling_rate *
                      mel_window_width / 1000),
            hop_length=int(global_sampling_rate *
                           mel_window_step / 1000),
            n_mels=mel_channels_count
        )
        mel_frames = mel_frames.astype(np.float32).T
        return mel_frames

    @staticmethod
    def fill_gaps(samples_per_voice_activity_window, is_window_contains_speech):
        structure_element = np.ones(
            samples_per_voice_activity_window + 1)
        return binary_dilation(is_window_contains_speech, structure_element)

    @staticmethod
    def save_mel(preprocessing_output_folder, mel_frames, audio_path):
        if not os.path.exists(preprocessing_output_folder):
            os.makedirs(preprocessing_output_folder)
        np.save(preprocessing_output_folder + "\\" +
                os.path.split(audio_path)[1].split(".")[0]+'.npy', mel_frames)

    @staticmethod
    def get_melspectrograms_for_training_iteration(current_training_iteration, mels_count_per_iteration, loaded_mels, global_frames_count, device):
        mels_start = current_training_iteration * mels_count_per_iteration
        mels_end = len(loaded_mels) if mels_start + mels_count_per_iteration > len(loaded_mels) else mels_start + mels_count_per_iteration
        training_mels = loaded_mels[mels_start: mels_end]
        training_frames = EncoderUtils.extract_frames_from_training_mels(global_frames_count, training_mels)
        return from_numpy(np.array(training_frames)).to(device)

    @staticmethod
    def extract_frames_from_training_mels(global_frames_count, training_mels):
        frames = []
        for mel in tqdm(training_mels, desc="Extracting training frames"):
            if mel.shape[0] == global_frames_count:
                mel_sample_start = 0
            else:
                mel_sample_start = np.random.randint(
                    0, mel.shape[0] - global_frames_count)
            mel_sample_end = mel_sample_start + global_frames_count
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
    def reshape_embeddings(global_frames_count, device, embeddings: Tensor):
        return embeddings.view((embeddings.shape[0], global_frames_count, INFERRED_DIMENSION)).to(device)

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
