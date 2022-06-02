import warnings
import os
import sys
import numpy as np
from statistics import mean
from torch import nn, optim
from torch import tensor, from_numpy, save, mean, norm, load, as_tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from encoder.utils import EncoderUtils
from encoder.encoder_config import EncoderConfiguration

TRAINING_MODE = True
SHOULD_SUPPRESS_NOISE = True

class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.load_configs()
        self.lstm = nn.LSTM(input_size=self.configs.mel_channels_count,
                            hidden_size=self.configs.hidden_layer_size,
                            num_layers=self.configs.layers_count,
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=self.configs.hidden_layer_size,
                                out_features=self.configs.embedding_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.similarity_weight = nn.Parameter(tensor([10.0])).to(device)
        self.similarity_bias = nn.Parameter(tensor([-5.0])).to(device)
        self.loss_function = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.configs.initial_learning_rate)
        self.current_training_iteration = 0
        self.device = device

    def load_configs(self):
        self.configs = EncoderConfiguration()
        self.configs.load()

        self.losses = []
        self.equal_error_rates = []

    def preprocess_dataset(self):
        if self.configs.should_preprocess:
            speakers_dirs = [self.configs.dataset_path + "\\" +
                            subdir for subdir in os.listdir(self.configs.dataset_path)]
            speakers_with_audios = [{speaker_dir: EncoderUtils.get_speaker_audios(
                speaker_dir)} for speaker_dir in speakers_dirs]

            for speaker in speakers_with_audios:
                for speaker_path, audios_paths in speaker.items():
                    for audio_path in tqdm(audios_paths, desc="Preprocessing "+speaker_path.split("\\")[-1]):
                        audio = EncoderUtils.get_audio(self.configs, audio_path)
                        mel_frames = EncoderUtils.convert_audio_to_mel_spectrogram_frames(self.configs, SHOULD_SUPPRESS_NOISE, audio)
                        if len(mel_frames) < self.configs.global_frames_count:
                            continue
                        else:
                            EncoderUtils.save_mel(self.configs, mel_frames, audio_path)

    def forward(self, samples, hidden_init=None):
        _, (hidden, _) = self.lstm(samples, hidden_init)
        unnormalized_embeddings = self.relu(self.linear(hidden[-1]))
        normalized_embeddings = unnormalized_embeddings / (norm(unnormalized_embeddings, dim=1, keepdim=True) + 1e-5)
        return normalized_embeddings

    def load_melspectrograms(self):
        self.loaded_mels = []
        for mel_file in tqdm(os.listdir(self.configs.preprocessing_output_folder), desc="Loading mel spectrograms"):
            self.loaded_mels.append(np.load(self.configs.preprocessing_output_folder + "\\" + mel_file))

    def do_training_iteration(self):
        training_mels = EncoderUtils.get_melspectrograms_for_training_iteration(self.configs, self.current_training_iteration, self.loaded_mels, self.device)
        if (list(training_mels.size())[0] > 0):
            loss = self.do_forward_pass(training_mels)
            self.do_backward_pass(loss)
            self.current_training_iteration += 1
            print("\nStep #" + str(self.current_training_iteration) + " --> Loss: " +
                  str(mean(as_tensor(self.losses)).numpy()) + "   Equal Error Rate: " + str(mean(as_tensor(self.equal_error_rates)).numpy()))
            if self.current_training_iteration % self.configs.checkpoint_frequency == 0:
                if not os.path.exists(self.configs.models_folder):
                    os.makedirs(self.configs.models_folder)
                save({"iteration": self.current_training_iteration, "model_state": self.state_dict(
                ), "optimizer_state": self.optimizer.state_dict()}, self.configs.models_folder + "\\encoder.pt")

    def do_forward_pass(self, training_mels):
        embeddings = self(training_mels.data)
        embeddings = EncoderUtils.reshape_embeddings(self.configs, self.device, embeddings)
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
        checkpoint_path = self.configs.models_folder + "\\encoder.pt"
        if os.path.exists(checkpoint_path):
            print("="*60+"\nStarting from encoder checkpoint!!\n"+"="*60)
            self.load_model(checkpoint_path, TRAINING_MODE)
        else:
            print("="*60+"\nNo encoder checkpoint, starting over!!\n"+"="*60)
        
        self.turn_on_training_mode()
        
        self.load_melspectrograms()
        for i in range(self.configs.training_iterations_count):
            print("="*60+"\nIteration #"+str(i+1)+":\n"+"="*15)
            self.do_training_iteration()

    def prepare_for_inference(self):
        checkpoint_path = self.configs.models_folder + "\\encoder.pt"
        self.load_model(checkpoint_path, not TRAINING_MODE)
        self.turn_off_training_mode()

    def get_embeddings_from_audio(self, audio_path):
        audio = EncoderUtils.get_audio(self.configs, audio_path)
        audio_partition_positions, mel_partition_positions = EncoderUtils.get_partitioning_positions(self.configs, len(audio))
        audio_end = audio_partition_positions[-1].stop
        if audio_end >= len(audio):
            audio = np.pad(audio, (0, audio_end - len(audio)), "constant")
        mel_frames = EncoderUtils.convert_audio_to_mel_spectrogram_frames(self.configs, not SHOULD_SUPPRESS_NOISE, audio)
        mel_paritions = np.array([mel_frames[partition] for partition in mel_partition_positions])
        mel_paritions = from_numpy(mel_paritions).to(self.device)
        partition_embeddings = self.forward(mel_paritions).detach().cpu().numpy()
        embeddings_avg = np.mean(partition_embeddings, axis=0)
        embeddings = embeddings_avg / np.linalg.norm(embeddings_avg, 2)
        return embeddings

    def load_model(self, checkpoint_path, training_mode):
        assert os.path.exists(checkpoint_path) == True, "encoder.pt model doesn't exist, please train first!!"
        checkpoint = load(checkpoint_path)
        self.load_state_dict(checkpoint["model_state"])
        if training_mode:
            self.current_training_iteration = checkpoint["iteration"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.optimizer.param_groups[0]["lr"] = self.configs.initial_learning_rate
    
    def turn_on_training_mode(self):
        self.train()
    
    def turn_off_training_mode(self):
        self.eval()