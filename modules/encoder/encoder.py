from utils import EncoderUtils
import warnings
import os
import sys
import yaml
import webrtcvad
import numpy as np
from statistics import mean
from torch import nn, optim
from torch import tensor, from_numpy, save, mean, norm, sum, zeros, load, as_tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


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
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.initial_learning_rate)
        self.current_training_iteration = 0
        self.device = device

    def load_configs(self):
        self.dataset_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\dataset")
        self.preprocessing_output_folder = os.path.join(
            os.path.dirname(__file__), "..\\..\\output\\preprocessing")
        self.models_folder = os.path.join(
            os.path.dirname(__file__), "..\\..\\models")

        with open(os.path.join(os.path.dirname(__file__), "..\\..\\configurations.yaml"), "r") as f:
            config = yaml.safe_load(f)

        self.global_sampling_rate = config["ENCODER"]["PREPROCESSING"]["SAMPLING_RATE"]
        self.average_amplitude_target_dBFS = config[
            "ENCODER"]["PREPROCESSING"]["AMPLITUDE_DBFS"]
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
        self.training_iterations_count = config[
            "ENCODER"]["TRAINING"]["TRAINING_ITERATIONS_COUNT"]
        self.mels_count_per_iteration = config[
            "ENCODER"]["TRAINING"]["MELS_PER_TRAINING_ITERATION"]
        self.initial_learning_rate = config["ENCODER"]["TRAINING"]["INIT_LEARNING_RATE"]

        self.losses = []
        self.equal_error_rates = []

    def preprocess_dataset(self):
        speakers_dirs = [self.dataset_path + "\\" +
                         subdir for subdir in os.listdir(self.dataset_path)]
        speakers_with_audios = [{speaker_dir: EncoderUtils.get_speaker_audios(
            speaker_dir)} for speaker_dir in speakers_dirs]

        for speaker in speakers_with_audios:
            for speaker_path, audios_paths in speaker.items():
                for audio_path in tqdm(audios_paths, desc="Preprocessing "+speaker_path.split("\\")[-1]):
                    audio = EncoderUtils.get_audio_with_correct_sampling_rate(self.global_sampling_rate, audio_path)
                    audio = EncoderUtils.normalize_amplitude(self.average_amplitude_target_dBFS, audio)
                    audio = EncoderUtils.trim_extra(self.samples_per_voice_activity_window, audio)
                    pcm = EncoderUtils.get_16_bit_pulse_code_modulation(audio)
                    voice_activity_detection = webrtcvad.Vad(mode=3)
                    is_window_contains_speech = EncoderUtils.detect_windows_containing_speech(
                        self.samples_per_voice_activity_window, self.global_sampling_rate, self.moving_average_width,
                        audio, pcm, voice_activity_detection)
                    is_window_contains_speech = EncoderUtils.fill_gaps(self.samples_per_voice_activity_window,
                                                                       is_window_contains_speech)
                    sample_positions = EncoderUtils.get_sample_positions(self.samples_per_voice_activity_window,
                                                                         is_window_contains_speech)
                    audio_samples = audio[sample_positions]
                    mel_frames = EncoderUtils.get_mel_frames(
                        self.global_sampling_rate, self.mel_window_width,
                        self.mel_window_step, self.mel_channels_count, audio_samples)
                    if len(mel_frames) < self.global_frames_count:
                        continue
                    else:
                        EncoderUtils.save_mel(self.preprocessing_output_folder, mel_frames, audio_path)

    def forward(self, samples, hidden_init=None):
        _, (hidden, _) = self.lstm(samples, hidden_init)
        unnormalized_embeddings = self.relu(self.linear(hidden[-1]))
        normalized_embeddings = unnormalized_embeddings / (norm(unnormalized_embeddings, dim=1, keepdim=True) + 1e-5)
        return normalized_embeddings

    def load_melspectrograms(self):
        self.loaded_mels = []
        for mel_file in tqdm(os.listdir(self.preprocessing_output_folder), desc="Loading mel spectrograms"):
            self.loaded_mels.append(np.load(self.preprocessing_output_folder + "\\" + mel_file))

    def do_training_iteration(self):
        training_mels = EncoderUtils.get_melspectrograms_for_training_iteration(
            self.current_training_iteration, self.mels_count_per_iteration, self.loaded_mels, 
            self.global_frames_count, self.device)
        if (list(training_mels.size())[0] > 0):
            loss = self.do_forward_pass(training_mels)
            self.do_backward_pass(loss)
            self.current_training_iteration += 1
            print("\nStep #" + str(self.current_training_iteration) + " --> Loss: " +
                  str(mean(as_tensor(self.losses)).numpy()) + "   Equal Error Rate: " + str(mean(as_tensor(self.equal_error_rates)).numpy()))
            if self.current_training_iteration % self.checkpoint_frequency == 0:
                if not os.path.exists(self.models_folder):
                    os.makedirs(self.models_folder)
                save({"iteration": self.current_training_iteration, "model_state": self.state_dict(
                ), "optimizer_state": self.optimizer.state_dict()}, self.models_folder + "\\encoder.pt")

    def do_forward_pass(self, training_mels):
        embeddings = self(training_mels.data)
        embeddings = EncoderUtils.reshape_embeddings(self.global_frames_count, self.device, embeddings)
        return self.loss(embeddings)

    def loss(self, embeddings):
        mels_count, samples_per_mel = embeddings.shape[:2]

        sim_matrix = EncoderUtils.calculate_similarity_matrix(self.similarity_weight, self.similarity_bias, self.device, embeddings).reshape((mels_count * samples_per_mel, mels_count))
        target_values = np.repeat(np.arange(mels_count), samples_per_mel)
        loss = self.loss_function(sim_matrix, from_numpy(target_values).long().to(self.device))
        self.losses.append(loss)

        equal_error_rate = EncoderUtils.calculate_equal_error_rate(mels_count, sim_matrix, target_values)
        self.equal_error_rates.append(equal_error_rate)

        return loss

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
