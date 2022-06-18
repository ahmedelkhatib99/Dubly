import os
import yaml
from dataclasses import dataclass

@dataclass
class SynthesizerConfig:
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
    batch_size:int 
    synthesis_batch_size: int
    sampling_rate: int
    rescale: bool 
    rescale_max: float 
    trim_silence: bool
    int16_max: int
    vad_window_length: int
    vad_moving_average_width:int 
    vad_max_silence_length: int

    # Tacotron Text-to-Speech (TTS)
    ref_level_db: int
    # Embedding dimension for the graphemes/phoneme inputs
    num_mels: int
    tts_embed_dims :int
    tts_encoder_dims :int
    tts_decoder_dims :int
    tts_postnet_dims :int
    tts_encoder_K :int
    tts_lstm_dims :int
    tts_postnet_K :int
    tts_num_highways :int
    tts_dropout: float
    tts_stop_threshold: float
    fmin:int
    fmax:int
    n_fft:int
    # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
    hop_length:int
    # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
    win_length:int
    min_level_db:int
    # Gradient explodes if too big, premature convergence if too small.
    max_abs_value: float
    # Filter coefficient to use if preemphasize is True
    preemphasis: float

    speaker_embedding_size:int              # Dimension for the speaker embedding

    def __init__(self):
        pass

    def load(self):
        
        with open(os.path.join(os.path.dirname(__file__), "..\\..\\configurations.yaml"), "r") as f:
            config = yaml.safe_load(f)
        self.batch_size = config["SYNTHESIZER"]["TRAINING"]["BATCH_SIZE"]
        self.synthesis_batch_size = config["SYNTHESIZER"]["TRAINING"]["SYNTHESIS_BATCH_SIZE"]
        self.sampling_rate = config["ENCODER"]["PREPROCESSING"]["SAMPLING_RATE"]
        self.rescale = config["SYNTHESIZER"]["PREPROCESSING"]["RESCALE"]
        self.rescale_max = config["SYNTHESIZER"]["PREPROCESSING"]["RESCALE_MAX"]
        self.trim_silence = config["SYNTHESIZER"]["PREPROCESSING"]["TRIM_SILENCE"]
        self.int16_max = config["SYNTHESIZER"]["PREPROCESSING"]["INT16_MAX"]
        self.vad_window_length = config["SYNTHESIZER"]["PREPROCESSING"]["VAD_WINDOW_LENGTH"]
        self.vad_moving_average_width = config["SYNTHESIZER"]["PREPROCESSING"]["VAD_MOVING_AVERAGE_WIDTH"]
        self.vad_max_silence_length = config["SYNTHESIZER"]["PREPROCESSING"]["VAD_MAX_SILENCE_LENGTH"]
        self.ref_level_db = config["SYNTHESIZER"]["TACTRON"]["REF_LEVEL_DB"]
        self.num_mels = config["SYNTHESIZER"]["TACTRON"]["NUM_MELS"]
        self.tts_embed_dims = config["SYNTHESIZER"]["TACTRON"]["EMBED_DIMS"]
        self.tts_encoder_dims = config["SYNTHESIZER"]["TACTRON"]["ENCODER_DIMS"]
        self.tts_decoder_dims = config["SYNTHESIZER"]["TACTRON"]["DECODER_DIMS"]
        self.tts_postnet_dims = config["SYNTHESIZER"]["TACTRON"]["POSTNET_DIMS"]
        self.tts_encoder_K = config["SYNTHESIZER"]["TACTRON"]["ENCODER_K"]
        self.tts_lstm_dims = config["SYNTHESIZER"]["TACTRON"]["LSTM_DIMS"]
        self.tts_postnet_K = config["SYNTHESIZER"]["TACTRON"]["POSTNET_K"]
        self.tts_num_highways = config["SYNTHESIZER"]["TACTRON"]["NUM_HIGHWAYS"]
        self.tts_dropout = config["SYNTHESIZER"]["TACTRON"]["DROPOUT"]
        self.tts_stop_threshold = config["SYNTHESIZER"]["TACTRON"]["STOP_THRESHOLD"]
        self.n_fft = config["SYNTHESIZER"]["TACTRON"]["N_FFT"]
        self.fmin = config["SYNTHESIZER"]["PREPROCESSING"]["FMIN"]
        self.fmax = config["SYNTHESIZER"]["PREPROCESSING"]["FMAX"]
        self.hop_length = config["SYNTHESIZER"]["PREPROCESSING"]["HOP_LENGTH"]
        self.win_length = config["SYNTHESIZER"]["PREPROCESSING"]["WIN_LENGTH"]
        self.min_level_db = config["SYNTHESIZER"]["PREPROCESSING"]["MIN_LEVEL_DB"]
        self.max_abs_value = config["SYNTHESIZER"]["PREPROCESSING"]["MAX_ABS_VALUE"]
        self.preemphasis = config["SYNTHESIZER"]["PREPROCESSING"]["PREEMPHASIS"]
        self.speaker_embedding_size = config["SYNTHESIZER"]["TRAINING"]["SPEAKER_EMBEDDING_SIZE"]

        