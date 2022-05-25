import warnings
import os
import yaml
import librosa
import struct
import webrtcvad
import numpy as np
from statistics import mean
from torch import embedding, nn, tensor, device, cuda, from_numpy, Tensor, save, optim, no_grad, mean, norm, sum, zeros, load, as_tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

INFERRED_DIMENSION = -1


class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.load_configs()
        self.lstm = nn.LSTM(input_size=self.mel_channels_count,
                            hidden_size=self.hidden_layer_size,
                            num_layers=self.layers_count,
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=self.hidden_layer_size,
                              out_features=self.embedding_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.similarity_weight = nn.Parameter(tensor([10.0])).to(device)
        self.similarity_bias = nn.Parameter(tensor([-5.0])).to(device)
        self.loss_function = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.initial_learning_rate)
        self.current_training_iteration = 0
        self.device = device

    def load_configs(self):
        self.dataset_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\dataset")
        self.preprocessing_output_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\output\\preprocessing")
        self.models_folder = os.path.join(
            os.path.dirname(__file__), "..\\..\\models")

        with open(os.path.join(os.path.dirname(__file__), "..\\..\\configurations.yaml"), "r") as f:
            config = yaml.safe_load(f)

        self.global_sampling_rate = config["ENCODER"]["PREPROCESSING"]["SAMPLING_RATE"]
        self.average_amplitude_target_dBFS = config["ENCODER"]["PREPROCESSING"]["AMPLITUDE_DBFS"]
        voice_activity_window_msecs = config["ENCODER"]["PREPROCESSING"]["VAD_WINDOW_MS"]
        self.samples_per_voice_activity_window = (
            voice_activity_window_msecs * self.global_sampling_rate) // 1000
        self.moving_average_width = config["ENCODER"]["PREPROCESSING"]["VAD_MOVING_AVG_WIDTH"]

        self.mel_window_width = config["ENCODER"]["MEL_SPECTROGRAM"]["MEL_WINDOW_WIDTH"]
        self.mel_window_step = config["ENCODER"]["MEL_SPECTROGRAM"]["MEL_WINDOW_STEP"]
        self.mel_channels_count = config["ENCODER"]["MEL_SPECTROGRAM"]["MEL_CHANNELS_COUNT"]
        self.global_frames_count = config["ENCODER"]["MEL_SPECTROGRAM"]["FRAMES_MIN_COUNT"]

        self.layers_count = config["ENCODER"]["MODEL"]["LAYERS_COUNT"]
        self.hidden_layer_size = config["ENCODER"]["MODEL"]["HIDDEN_LAYER_SIZE"]
        self.embedding_size = config["ENCODER"]["MODEL"]["EMBEDDING_SIZE"]

        self.checkpoint_frequency = config["ENCODER"]["TRAINING"]["CHECKPOINT_FREQUENCY"]
        self.training_iterations_count = config["ENCODER"]["TRAINING"]["TRAINING_ITERATIONS_COUNT"]
        self.mels_count_per_iteration = config["ENCODER"]["TRAINING"]["MELS_PER_TRAINING_ITERATION"]
        self.initial_learning_rate = config["ENCODER"]["TRAINING"]["INIT_LEARNING_RATE"]

        self.losses = []
        self.equal_error_rates = []

    def preprocess_dataset(self):
        speakers_dirs = [self.dataset_path + "\\" +
                         subdir for subdir in os.listdir(self.dataset_path)]
        speakers_with_audios = [{speaker_dir: self.get_speaker_audios(
            speaker_dir)} for speaker_dir in speakers_dirs]

        for speaker in speakers_with_audios:
            for speaker_path, audios_paths in speaker.items():
                for audio_path in tqdm(audios_paths, desc="Preprocessing "+speaker_path.split("\\")[-1]):
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
                    if len(mel_frames) < self.global_frames_count:
                        continue
                    else:
                        self.save_mel(mel_frames, audio_path)

    def forward(self, samples, hidden_init=None):
        _, (hidden, _) = self.lstm(samples, hidden_init)
        unnormalized_embeddings = self.relu(self.linear(hidden[-1]))
        normalized_embeddings = unnormalized_embeddings / (norm(unnormalized_embeddings, dim=1, keepdim=True) + 1e-5)
        return normalized_embeddings

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

    def get_melspectrograms_for_training_iteration(self):
        mels_start = self.current_training_iteration * self.mels_count_per_iteration
        mels_end = len(self.loaded_mels) if mels_start + self.mels_count_per_iteration > len(
            self.loaded_mels) else mels_start + self.mels_count_per_iteration
        training_mels = self.loaded_mels[mels_start: mels_end]
        training_frames = self.extract_frames_from_training_mels(training_mels)
        return from_numpy(np.array(training_frames)).to(self.device)

    def extract_frames_from_training_mels(self, training_mels):
        frames = []
        for mel in tqdm(training_mels, desc="Extracting training frames"):
            if mel.shape[0] == self.global_frames_count:
                mel_sample_start = 0
            else:
                mel_sample_start = np.random.randint(
                    0, mel.shape[0] - self.global_frames_count)
            mel_sample_end = mel_sample_start + self.global_frames_count
            frames.append(mel[mel_sample_start: mel_sample_end])
        return frames

    def load_melspectrograms(self):
        self.loaded_mels = []
        for mel_file in tqdm(os.listdir(self.preprocessing_output_path), desc="Loading mel spectrograms"):
            self.loaded_mels.append(
                np.load(self.preprocessing_output_path + "\\" + mel_file))

    def reshape_embeddings(self, embeddings: Tensor):
        return embeddings.view((embeddings.shape[0], self.global_frames_count, INFERRED_DIMENSION)).to(self.device)

    def do_training_iteration(self):
        training_mels = self.get_melspectrograms_for_training_iteration()
        if (list(training_mels.size())[0] > 0):
            loss = self.do_forward_pass(training_mels)
            self.do_backward_pass(loss)
            self.current_training_iteration += 1
            print("\nStep #" + str(self.current_training_iteration) + " --> Loss: " +
                str(mean(as_tensor(self.losses)).numpy()) + "   Equal Error Rate: " + str(mean(as_tensor(self.equal_error_rates)).numpy()))
            if self.current_training_iteration % self.checkpoint_frequency == 0:
                save({"iteration": self.current_training_iteration, "model_state": self.state_dict(
                ), "optimizer_state": self.optimizer.state_dict()}, self.models_folder + "\\encoder.pt")

    def do_forward_pass(self, training_mels):
        embeddings = self(training_mels.data)
        embeddings = self.reshape_embeddings(embeddings)
        return self.loss(embeddings)

    def loss(self, embeddings):
        mels_count, samples_per_mel = embeddings.shape[:2]

        sim_matrix = self.calculate_similarity_matrix(embeddings).reshape(
            (mels_count * samples_per_mel, mels_count))
        target_values = np.repeat(np.arange(mels_count), samples_per_mel)
        loss = self.loss_function(sim_matrix, from_numpy(
            target_values).long().to(self.device))
        self.losses.append(loss)

        equal_error_rate = self.calculate_equal_error_rate(
            mels_count, sim_matrix, target_values)
        self.equal_error_rates.append(equal_error_rate)

        return loss

    def calculate_similarity_matrix(self, embeddings):

        mels_count, samples_per_mel = embeddings.shape[:2]

        mels_centroids = mean(embeddings, dim=1, keepdim=True)
        mels_centroids = mels_centroids.clone(
        ) / (norm(mels_centroids, dim=2, keepdim=True) + 1e-5)

        samples_centroids = (
            sum(embeddings, dim=1, keepdim=True) - embeddings) / (samples_per_mel - 1)
        samples_centroids = samples_centroids.clone(
        ) / (norm(samples_centroids, dim=2, keepdim=True) + 1e-5)

        similarity_matrix = zeros(mels_count, samples_per_mel, mels_count).to(self.device)
        all_mels_positions = 1 - np.eye(mels_count, dtype=np.int)
        for mel_index in range(mels_count):
            mel_positions = np.where(all_mels_positions[mel_index])[0]
            similarity_matrix[mel_positions, :, mel_index] = (embeddings[mel_positions] * mels_centroids[mel_index]).sum(dim=2)
            similarity_matrix[mel_index, :, mel_index] = (embeddings[mel_index] * samples_centroids[mel_index]).sum(dim=1)
        
        return similarity_matrix * self.similarity_weight + self.similarity_bias

    def calculate_equal_error_rate(self, mels_count, sim_matrix, ground_truth):
        with no_grad():
            def inv_argmax(i): return np.eye(1, mels_count, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            false_positive_rates, true_positive_rate, _ = roc_curve(
                labels.flatten(), preds.flatten())
            equal_error_rate = brentq(
                lambda x: 1. - x - interp1d(false_positive_rates, true_positive_rate)(x), 0.0, 1.0)
        return equal_error_rate

    def do_backward_pass(self, loss):
        self.zero_grad()
        loss.backward()
        self.update_gradients()
        self.optimizer.step()

    def update_gradients(self):
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def start_training(self):
        checkpoint_path = self.models_folder + "\\encoder.pt"
        if os.path.exists(checkpoint_path):
            print("="*60+"\nStarting from encoder checkpoint!!\n"+"="*60)
            checkpoint = load(checkpoint_path)
            self.current_training_iteration = checkpoint["iteration"]
            self.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.optimizer.param_groups[0]["lr"] = self.initial_learning_rate
        else:
            print("="*60+"\nNo encoder checkpoint, starting over!!\n"+"="*60)
        self.load_melspectrograms()
        for i in range(self.training_iterations_count):
            print("="*60+"\nIteration #"+str(i+1)+":\n"+"="*15)
            self.do_training_iteration()
    
    def turn_on_training_mode(self):
        self.train()


device = device("cuda" if cuda.is_available() else "cpu")
encoder = Encoder(device)
encoder.preprocess_dataset()
encoder.turn_on_training_mode()
encoder.start_training()