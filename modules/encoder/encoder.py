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

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from utils import EncoderUtils

TRAINING_MODE = True
SHOULD_SUPPRESS_NOISE = True

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
                    audio = self.get_audio(self.global_sampling_rate, self.average_amplitude_target_dBFS, self.samples_per_voice_activity_window, audio_path)
                    mel_frames = self.convert_audio_to_mel_spectrogram_frames(self.global_sampling_rate, self.samples_per_voice_activity_window, 
                                                                              self.moving_average_width, self.mel_window_step, 
                                                                              self.mel_window_width, self.mel_channels_count, 
                                                                              SHOULD_SUPPRESS_NOISE, audio)
                    if len(mel_frames) < self.global_frames_count:
                        continue
                    else:
                        EncoderUtils.save_mel(self.preprocessing_output_folder, mel_frames, audio_path)

    def convert_audio_to_mel_spectrogram_frames(self, global_sampling_rate, samples_per_voice_activity_window, moving_average_width, mel_window_step, mel_window_width, mel_channels_count, should_suppress_noise, audio):
        audio_samples = audio
        if should_suppress_noise:
            pcm = EncoderUtils.get_16_bit_pulse_code_modulation(audio)
            voice_activity_detection = webrtcvad.Vad(mode=3)
            is_window_contains_speech = EncoderUtils.detect_windows_containing_speech(
                            samples_per_voice_activity_window, global_sampling_rate, moving_average_width,
                            audio, pcm, voice_activity_detection)
            is_window_contains_speech = EncoderUtils.fill_gaps(samples_per_voice_activity_window,
                                                                        is_window_contains_speech)
            sample_positions = EncoderUtils.get_sample_positions(samples_per_voice_activity_window,
                                                                            is_window_contains_speech)
            audio_samples = audio[sample_positions]
        mel_frames = EncoderUtils.get_mel_frames(
                        global_sampling_rate, mel_window_width,
                        mel_window_step, mel_channels_count, audio_samples)
            
        return mel_frames

    def get_audio(self, global_sampling_rate, average_amplitude_target_dBFS, samples_per_voice_activity_window, audio_path):
        audio = EncoderUtils.get_audio_with_correct_sampling_rate(global_sampling_rate, audio_path)
        audio = EncoderUtils.normalize_amplitude(average_amplitude_target_dBFS, audio)
        audio = EncoderUtils.trim_extra(samples_per_voice_activity_window, audio)
        return audio

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
            self.load_model(checkpoint_path, TRAINING_MODE)
        else:
            print("="*60+"\nNo encoder checkpoint, starting over!!\n"+"="*60)
        
        self.turn_on_training_mode()
        
        self.load_melspectrograms()
        for i in range(self.training_iterations_count):
            print("="*60+"\nIteration #"+str(i+1)+":\n"+"="*15)
            self.do_training_iteration()

    def prepare_for_inference(self):
        checkpoint_path = self.models_folder + "\\encoder.pt"
        self.load_model(checkpoint_path, not TRAINING_MODE)
        self.turn_off_training_mode()
    
    def get_partitioning_positions(self, audio_length):
        samples_per_frame = int((self.global_sampling_rate * self.mel_window_step / 1000))
        frames_count = int(np.ceil((audio_length + 1) / samples_per_frame))
        step_size = max(int(np.round(self.global_frames_count * 0.5)), 1)
        steps_count = max(1, frames_count - self.global_frames_count + step_size + 1)
        audio_partition_positions = []
        mel_partition_positions = []
        for i in range(0, steps_count, step_size):
            mel_partition = np.array([i, i + self.global_frames_count])
            audio_partition = mel_partition * samples_per_frame
            mel_partition_positions.append(slice(*mel_partition))
            audio_partition_positions.append(slice(*audio_partition))
        last_audio_parition_position = audio_partition_positions[-1]
        coverage = (audio_length - last_audio_parition_position.start) / (last_audio_parition_position.stop - last_audio_parition_position.start)
        if coverage < 0.75 and len(mel_partition_positions) > 1:
            mel_partition_positions = mel_partition_positions[:-1]
            audio_partition_positions = audio_partition_positions[:-1]

        return audio_partition_positions, mel_partition_positions

    def get_embeddings_from_audio(self, audio_path):
        audio = self.get_audio(self.global_sampling_rate, self.average_amplitude_target_dBFS, self.samples_per_voice_activity_window, audio_path)
        audio_partition_positions, mel_partition_positions = self.get_partitioning_positions(len(audio))
        audio_end = audio_partition_positions[-1].stop
        if audio_end >= len(audio):
            audio = np.pad(audio, (0, audio_end - len(audio)), "constant")
        mel_frames = self.convert_audio_to_mel_spectrogram_frames(self.global_sampling_rate, self.samples_per_voice_activity_window, 
                                                                  self.moving_average_width, self.mel_window_step, 
                                                                  self.mel_window_width, self.mel_channels_count, 
                                                                  not SHOULD_SUPPRESS_NOISE, audio)
        mel_paritions = np.array([mel_frames[partition] for partition in mel_partition_positions])
        mel_paritions = from_numpy(mel_paritions).to(self.device)
        partition_embeddings = self.forward(mel_paritions).detach().cpu().numpy()
        embeddings_avg = np.mean(partition_embeddings, axis=0)
        embeddings = embeddings_avg / np.linalg.norm(embeddings_avg, 2)
        return embeddings

    def load_model(self, checkpoint_path, training_mode):
        assert os.path.exists(checkpoint_path) == True, "encoder.pt model doesn't exist, please train first!!"
        checkpoint = load(checkpoint_path, self.device)
        self.load_state_dict(checkpoint["model_state"])
        if training_mode:
            self.current_training_iteration = checkpoint["iteration"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.optimizer.param_groups[0]["lr"] = self.initial_learning_rate
    
    def turn_on_training_mode(self):
        self.train()
    
    def turn_off_training_mode(self):
        self.eval()