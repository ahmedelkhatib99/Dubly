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

INFER_EMBEDDING_SIZE = -1
MOST_AGRESSIVE = 3


class EncoderUtils:
    @staticmethod
    def get_speaker_audios(speaker_dir):
        """Returns the paths of audio files inside the directory of a certain speaker
        """
        speaker_audios = []
        for subfolder in os.listdir(speaker_dir):
            for file in os.listdir(speaker_dir + "\\" + subfolder):
                if os.path.isfile(speaker_dir + "\\" + subfolder + "\\" + file) and (file.endswith(".mp3") or file.endswith(".flac")):
                    speaker_audios.append(speaker_dir + "\\" + subfolder + "\\" + file)
        return speaker_audios
        # return [speaker_dir + "\\" + file for file in os.listdir(speaker_dir) if os.path.isfile(speaker_dir + "\\" + file) and file.endswith(".mp3")]

    @staticmethod
    def convert_audio_to_melspectrogram_frames(configs: EncoderConfiguration, should_suppress_noise, audio):
        """Returns audio converted from waveform to mel spectogram to be easier to handle.
        
        Set should_suppress_noise to true especially while training to improve it.\n
        """

        audio_samples = audio
        if should_suppress_noise:
            audio_samples = EncoderUtils.suppress_noise(configs, audio)
        mel_frames = EncoderUtils.get_mel_frames(configs, audio_samples)
        return mel_frames

    @staticmethod
    def suppress_noise(configs: EncoderConfiguration, audio):
        """Returns voice without noise or unnecessary silence

        To suppress noise the following is done:
            1- Put the correctly sampled wave in 16 bit integer representation.\n
            2- Detect windows that contain voice and represent them with flags.\n
            3- Fill silence gaps that are within the acceptable range.\n
            4- Sample the audio at the positions that are considered to contain voice only.\n
        """
        voice_activity_detection = webrtcvad.Vad(mode=MOST_AGRESSIVE)
        pcm = EncoderUtils.get_16_bit_pulse_code_modulation(audio)
        is_window_contains_speech = EncoderUtils.detect_windows_containing_speech(configs, audio, pcm, voice_activity_detection)
        is_window_contains_speech = EncoderUtils.fill_gaps(configs, is_window_contains_speech)
        sample_positions = EncoderUtils.get_sample_positions(configs, is_window_contains_speech)
        audio_samples = audio[sample_positions]
        return audio_samples

    @staticmethod
    def get_audio(configs: EncoderConfiguration, audio_path):
        """Return float32 representation of the audio file
        
        Audio file sampling rate is adjusted to the global sampling rate in the configuration file,
        then it is normalized and the length is adjusted to be an integer multiple of the window size
        in the configuration file
        """
        
        audio = EncoderUtils.get_audio_with_correct_sampling_rate(configs, audio_path)
        audio = EncoderUtils.normalize_amplitude(configs, audio)
        audio = EncoderUtils.trim_extra(configs, audio)
        return audio

    @staticmethod
    def get_audio_with_correct_sampling_rate(configs: EncoderConfiguration, audio_path):
        """Returns a float32 representation of the audio file
        
        Sampling rate is checked against the global sampling rate configuration,
        if it doesn't match it, the audio is resampled
        """
        
        audio, sampling_rate = librosa.load(audio_path, sr=None)
        if sampling_rate is not None and sampling_rate != configs.global_sampling_rate:
            audio = librosa.resample(audio, sampling_rate, configs.global_sampling_rate)
        return audio

    @staticmethod
    def normalize_amplitude(configs: EncoderConfiguration, audio):
        """Returns audio with normalized amplitude

        Amplitude error is calculated in decibles relative to full scale and if the error is above 0,
        this indicates that the mean value of the audio is below the required average amplitude
        thus the audio is normalized to meet the required average amplitude value
        """
        
        amplitude_error_dbFS = configs.average_amplitude_target_dBFS - 10 * np.log10(np.mean(audio ** 2))
        if amplitude_error_dbFS > 0:
            audio = audio * (10 ** (amplitude_error_dbFS / 20))
        return audio

    @staticmethod
    def trim_extra(configs: EncoderConfiguration, audio):
        """Returns audio that is an integer multiple of the number of samples equivalent to the voice window size in the configuration file
        """
        
        return audio[:len(audio) - (len(audio) % configs.samples_per_voice_activity_window)]

    @staticmethod
    def get_16_bit_pulse_code_modulation(audio):
        """Returns an audio representation converted from float32 to int16
        
        Audio is first converted from float32 to python float, then rounded and converted to int16"""
        return struct.pack("%dh" % len(audio), *(np.round(audio * ((2 ** 15) - 1))).astype(np.int16))

    @staticmethod
    def detect_windows_containing_speech(configs: EncoderConfiguration, audio, pcm, voice_activity_detection):
        """Returns an array of boolean flags indicating position of windows containing speech
        
        pcm has to be a 16 bit integer representation of the audio sampled at the global sampling rate
        NB: global sampling rate is required to be 8000, 16000, 32000 or 48000Hz. In our configuration it is 16000
        """
        
        is_contains_speech = []
        for start in range(0, len(audio), configs.samples_per_voice_activity_window):
            end = start + configs.samples_per_voice_activity_window
            is_contains_speech.append(voice_activity_detection.is_speech(pcm[start * 2: end * 2], sample_rate=configs.global_sampling_rate))
        is_contains_speech = np.array(is_contains_speech)
        is_contains_speech = EncoderUtils.smooth_windows_containing_detected_speech(configs, is_contains_speech)
        is_contains_speech = np.round(is_contains_speech).astype(np.bool)
        return is_contains_speech

    @staticmethod
    def smooth_windows_containing_detected_speech(configs: EncoderConfiguration, is_window_contains_speech):
        """Returns a smoothed value of the voice detection flags
        
        Applying a moving window average technique to consider the effect of neighboring windows in the detection
        """
        
        padded_is_window_contains_speech = np.concatenate((np.zeros((configs.moving_average_width - 1) // 2), is_window_contains_speech, np.zeros(configs.moving_average_width // 2)))
        smoothed_is_window_contains_speech = np.cumsum(padded_is_window_contains_speech, dtype=float)
        smoothed_is_window_contains_speech[configs.moving_average_width:] = smoothed_is_window_contains_speech[configs.moving_average_width:] - smoothed_is_window_contains_speech[:-configs.moving_average_width]
        return smoothed_is_window_contains_speech[configs.moving_average_width - 1:] / configs.moving_average_width

    @staticmethod
    def get_sample_positions(configs: EncoderConfiguration, is_window_contains_speech):
        """Return masks (i.e. windows of ones) corresponding to windows that are detected to contain speech
        """
        
        return np.repeat(is_window_contains_speech, configs.samples_per_voice_activity_window)

    @staticmethod
    def get_mel_frames(configs: EncoderConfiguration, audio_samples):
        """Return melspectrogram frames representing the waveforms of the audio samples
        """
        
        mel_frames = librosa.feature.melspectrogram(
            audio_samples,
            configs.global_sampling_rate,
            n_fft=int(configs.global_sampling_rate * (configs.mel_window_width / 1000)),
            hop_length=int(configs.global_sampling_rate * (configs.mel_window_step / 1000)),
            n_mels=configs.mel_channels_count
        )
        mel_frames = mel_frames.astype(np.float32).T
        return mel_frames

    @staticmethod
    def fill_gaps(configs: EncoderConfiguration, is_window_contains_speech):
        """Returns dilated flags to tolerate silences withing the acceptable range
        """
        
        structure_element = np.ones(
            configs.max_silence + 1)
        return binary_dilation(is_window_contains_speech, structure_element)

    @staticmethod
    def save_mel(configs: EncoderConfiguration, mel_frames, audio_path, speaker_path):
        """Saves melspectrogram as a binary file in the processing output folder and under the folder of speaker name
        """
        output_folder = configs.preprocessing_output_folder + "\\" + speaker_path.split("\\")[-1]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        np.save(output_folder + "\\" + os.path.split(audio_path)[1].split(".")[0]+'.npy', mel_frames)

    @staticmethod
    def get_melspectrograms_for_training_iteration(configs: EncoderConfiguration, current_training_iteration, device):
        """Returns a batch of preprocessed melspectrograms to be used in training iteration
        
        The training batch consists of a number of speakers equal to that in the configuration file,
        for each speaker a number of melspectrograms equal to that in the configuration file is chosen randomly
        form the melspectrograms (corresponding to the preprocessed audios of the speaker) under the folder named after the speaker
        """
        
        speakers = os.listdir(configs.preprocessing_output_folder)
        speakers_start = (current_training_iteration * configs.speakers_count_per_iteration) % len(speakers)
        speakers_end = len(speakers) if speakers_start + configs.speakers_count_per_iteration > len(speakers) else speakers_start + configs.speakers_count_per_iteration
        training_speakers = speakers[speakers_start: speakers_end]
        training_mels = []
        for speaker in tqdm(training_speakers, desc="Loading melspectrograms"):
            speaker_mels = os.listdir(configs.preprocessing_output_folder + "\\" + speaker)
            for _ in range(configs.mels_count_per_speaker):
                choice = configs.preprocessing_output_folder + "\\" + speaker + "\\" + random.choice(speaker_mels)
                training_mels.append(np.load(choice))
        training_frames = EncoderUtils.extract_frames_from_training_mels(configs, training_mels)
        return from_numpy(np.array(training_frames)).to(device)

    @staticmethod
    def extract_frames_from_training_mels(configs: EncoderConfiguration, training_mels):
        """Returns the frames corresponding to the melspectrograms of the training batch
        
        If the number of frames in the melspectrogram exceeds the global frames count in the configuration file,
        a sample containing the required number of frames is choosen randomly to represent this frame
        """
        
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
        """Return the value of the similarity matrix which is the comparison of all embeddings against every speaker centroid

        For accuracy, when the embedding is removed from the centroid when compared with its own speaker to avoid biased/inaccurate results
        Speaker centroid is the normalized mean of the embeddings corresponding to the speaker's melspectrograms in the training batch.\n
        Similarity matrix is simply the result of dot products of all embeddings against every speaker centroid.
        """
        
        speakers_count, mels_count_per_speaker = embeddings.shape[:2]
        
        speakers_centroids = mean(embeddings, dim=1, keepdim=True)
        speakers_centroids = speakers_centroids.clone() / (norm(speakers_centroids, dim=2, keepdim=True) + 1e-5)
        

        centroids_excluding_speaker_mels = (sum(embeddings, dim=1, keepdim=True) - embeddings) / (mels_count_per_speaker - 1)
        centroids_excluding_speaker_mels = centroids_excluding_speaker_mels.clone() / (norm(centroids_excluding_speaker_mels, dim=2, keepdim=True) + 1e-5)

        similarity_matrix = zeros(speakers_count, mels_count_per_speaker, speakers_count).to(device)
        speakers_excluding_current = 1 - np.eye(speakers_count, dtype=np.int)
        for speaker_index in range(speakers_count):
            other_speakers_indices = np.where(speakers_excluding_current[speaker_index])[0]
            similarity_matrix[other_speakers_indices, :, speaker_index] = (embeddings[other_speakers_indices] * speakers_centroids[speaker_index]).sum(dim=2)
            similarity_matrix[speaker_index, :, speaker_index] = (embeddings[speaker_index] * centroids_excluding_speaker_mels[speaker_index]).sum(dim=1)

        return similarity_matrix * similarity_weight + similarity_bias

    @staticmethod
    def reshape_embeddings(configs: EncoderConfiguration, device, embeddings: Tensor):
        """Returns embeddings reshaped to be able to use it in calculating loss function

        Dimensions:\n
            1st (rows) --> speakers\n
            2nd (columns) --> melspectrograms\n
            3rd (depth of each cell) --> embedding size
        """
        
        return embeddings.view((configs.speakers_count_per_iteration, configs.mels_count_per_speaker, INFER_EMBEDDING_SIZE)).to(device)

    @staticmethod
    def calculate_equal_error_rate(speakers_count, sim_matrix, ground_truth):
        """Return Equal Error Rate (EER) which evaluates the accuracy of our embedding generation

        It gives insight about the value of the false acceptance rate when it equals false rejection rate
        """
        
        with no_grad():
            actual_labels = np.array([np.eye(1, speakers_count, original_value, dtype=np.int)[0] for original_value in ground_truth])
            predicted_labels = sim_matrix.detach().cpu().numpy()

            false_positive_rates, true_positive_rate, _ = roc_curve(actual_labels.flatten(), predicted_labels.flatten())
            equal_error_rate = brentq(lambda x: 1. - x - interp1d(false_positive_rates, true_positive_rate)(x), 0.0, 1.0)
        return equal_error_rate
    
    @staticmethod
    def get_partitioning_positions(configs: EncoderConfiguration, audio_length):
        """Returns positions of partitioning both the audio waveform and the mel spectrogram
        
        Partitioning improves the performance and the accuracy of inference.\n
        Each partition overlaps with half of its preceeding and following paritions. Paritions cover at least 75 percent of the audio
        """
        
        samples_per_frame, step_size, steps_count = EncoderUtils.get_partitioning_configurations(configs, audio_length)
        audio_partition_positions = []
        mel_partition_positions = []
        for i in range(0, steps_count, step_size):
            mel_partition = np.array([i, i + configs.global_frames_count])
            audio_partition = mel_partition * samples_per_frame
            mel_partition_positions.append(slice(*mel_partition))
            audio_partition_positions.append(slice(*audio_partition))
        coverage = EncoderUtils.calculate_coverage_until_second_last_parition(audio_length, audio_partition_positions)
        if coverage < 0.75 and len(mel_partition_positions) > 1:
            # if it is already 0.75 or more, that's enough and the last parition is not required. This can optimize in the inference performance.
            # but less than 0.75 is not accepted and the last parition is included to pass the threshold.
            audio_partition_positions = audio_partition_positions[:-1]
            mel_partition_positions = mel_partition_positions[:-1]
        return audio_partition_positions, mel_partition_positions

    @staticmethod
    def get_partitioning_configurations(configs: EncoderConfiguration, audio_length):
        """Return number of samples per frame, step size and count of steps
        
        Number of samples per frame is calculated as a function of the global sampling rate and the window step size in the configurations.
        The step size is calculated as half the global frames count in an audio so that each window contains a number of frames equal to the global frames count
        in which half of them overlap with the previous window and the other half overlaps with the next window. Steps count is calculated so that it is at least
        one step to ensure having at least one window that satisfies the global frames count configuration.
        """
        
        samples_per_frame = int((configs.global_sampling_rate * (configs.mel_window_step / 1000)))
        total_frames_count = int(np.ceil((audio_length + 1) / samples_per_frame))
        step_size = max(int(np.round(configs.global_frames_count * 0.5)), 1)
        steps_count = max(1, total_frames_count - configs.global_frames_count + step_size + 1)
        return samples_per_frame,step_size,steps_count

    @staticmethod
    def calculate_coverage_until_second_last_parition(audio_length, audio_partition_positions):
        """Return the percenatge of the audio that is covered by the calculated paritions from the first parition up until the second last partition
        """
        
        last_audio_parition_position = audio_partition_positions[-1]
        coverage = (audio_length - last_audio_parition_position.start) / (last_audio_parition_position.stop - last_audio_parition_position.start)
        return coverage
