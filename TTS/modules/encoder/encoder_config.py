import os
import yaml
from dataclasses import dataclass

@dataclass
class EncoderConfiguration:
    dataset_path: str
    preprocessing_output_folder: str
    models_folder: str
    should_preprocess: bool
    global_sampling_rate: int
    average_amplitude_target_dBFS: int
    max_silence: int
    samples_per_voice_activity_window: int
    moving_average_width: int
    mel_window_width: int
    mel_window_step: int
    mel_channels_count: int
    global_frames_count: int
    layers_count: int
    hidden_layer_size: int
    embedding_size: int
    checkpoint_frequency: int
    training_iterations_count: int
    speakers_count_per_iteration: int
    mels_count_per_speaker: int
    initial_learning_rate: int
    
    def __init__(self):
        pass

    def load(self):
        self.dataset_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\dataset")
        self.preprocessing_output_folder = os.path.join(
            os.path.dirname(__file__), "..\\..\\output\\preprocessing")
        self.models_folder = os.path.join(
            os.path.dirname(__file__), "..\\..\\models")
        with open(os.path.join(os.path.dirname(__file__), "..\\..\\configurations.yaml"), "r") as f:
            config = yaml.safe_load(f)
        self.should_preprocess = self.global_sampling_rate = config["ENCODER"]["PREPROCESSING"]["SHOULD_PREPROCESS"]
        self.global_sampling_rate = config["ENCODER"]["PREPROCESSING"]["SAMPLING_RATE"]
        self.average_amplitude_target_dBFS = config[
            "ENCODER"]["PREPROCESSING"]["AMPLITUDE_DBFS"]
        voice_activity_window_msecs = config["ENCODER"]["PREPROCESSING"]["VAD_WINDOW_MS"]
        self.max_silence = config["ENCODER"]["PREPROCESSING"]["MAX_SILENCE"]
        self.samples_per_voice_activity_window = (voice_activity_window_msecs * self.global_sampling_rate) // 1000
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
        self.speakers_count_per_iteration = config["ENCODER"]["TRAINING"]["SPEAKERS_PER_ITERATION"]
        self.mels_count_per_speaker = config["ENCODER"]["TRAINING"]["MELS_PER_SPEAKER"]
        self.initial_learning_rate = config["ENCODER"]["TRAINING"]["INIT_LEARNING_RATE"]