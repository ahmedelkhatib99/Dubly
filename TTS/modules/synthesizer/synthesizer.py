
from pathlib import Path
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage.morphology import binary_dilation
import struct
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..\\.."))

from modules.synthesizer.tactron import Tacotron
import torch
import torch.nn.functional as F
from torch import optim
from collections import defaultdict
from modules.synthesizer.symbols import int_to_text, text_to_int, symbols

try:
    import webrtcvad
except:
    print("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad = None


class Synthesizer:

    batch_size = 100
    synthesis_batch_size = 16

    SAMPLING_RATE = 16000
    RESCALE = True
    RESCALE_MAX = 0.9
    TRIM_SILENCE = True

    int16_max = (2 ** 15) - 1

    vad_window_length = 30  # ms
    vad_moving_average_width = 8
    vad_max_silence_length = 6

    # Tacotron Text-to-Speech (TTS)
    # Embedding dimension for the graphemes/phoneme inputs
    num_mels = 80
    tts_embed_dims = 512
    tts_encoder_dims = 256
    tts_decoder_dims = 128
    tts_postnet_dims = 512
    tts_encoder_K = 5
    tts_lstm_dims = 1024
    tts_postnet_K = 5
    tts_num_highways = 4
    tts_dropout = 0.5
    tts_stop_threshold = -3.4

    n_fft = 800
    # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
    hop_length = 200
    # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
    win_length = 800
    min_level_db = -100
    # Gradient explodes if too big, premature convergence if too small.
    max_abs_value = 4.
    # Filter coefficient to use if preemphasize is True
    preemphasis = 0.97

    speaker_embedding_size = 256               # Dimension for the speaker embedding


    def __init__(self):
        pass

    def preprocess(self, dataset_root, out_dir):

        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir()
            out_dir.joinpath("mels").mkdir()
            out_dir.joinpath("audio").mkdir()

        # Create a metadata file
        metadata_fpath = out_dir.joinpath("train.txt")
        metadata_file = metadata_fpath.open("w", encoding="utf-8")

        wav_embed_tuples = []
        dataset = Path(dataset_root)

        # Load records text
        records_text = defaultdict(lambda: '')
        for line in open(dataset.joinpath("text.tsv"), encoding="utf-8"):
            line = line.strip().split("\t")

            records_text[line[1]] = line[2]

        print(records_text)
        extensions = ["*.wav", "*.flac", "*.mp3"]
        for extension in extensions:
            for audio_path in dataset.glob(extension):
                print(str(audio_path))

                # Read the audio file and Resample and Rescale if needed
                wav, _ = librosa.load(str(audio_path), sr=self.SAMPLING_RATE)
                if self.RESCALE:
                    wav = wav/np.abs(wav).max() * self.RESCALE_MAX

                # Produce utterance for audio and write it to the file
                audio_name = str(audio_path.with_suffix("").name)
                mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % audio_name)
                wav_fpath = out_dir.joinpath(
                    "audio", "audio-%s.npy" % audio_name)

                # Trim silence
                if self.TRIM_SILENCE:
                    wav = self.trim_long_silences(wav)

                # Calculate Write the audio and mel spectrograms
                mel = self.mel_spectrogram(wav)

                # Write the spectrogram, embed and audio to disk
                np.save(mel_fpath, mel.T, allow_pickle=False)
                np.save(wav_fpath, wav, allow_pickle=False)

                # print()
                print(str(wav_fpath.name), str(mel_fpath.name),
                      "embed-%s.npy" % audio_name)
                wav_embed_tuples.append(
                    (wav_fpath.name, "embed-%s.npy" % audio_name))
                metadata_file.write("%s|%s|%s|%s\n" % (str(wav_fpath.name), str(
                    mel_fpath.name), "embed-%s.npy" % audio_name, records_text[audio_path.name]))

        metadata_file.close()

        # Generate Embeddings for training

        wav_dir = out_dir.joinpath("audio")
        embed_dir = out_dir.joinpath("embed")
        embed_dir.mkdir(exist_ok=True)

        fpaths = []
        for wav_fpath, embed_fname in wav_embed_tuples:
            wav_fpath = wav_dir.joinpath(wav_fpath)
            embed_fpath = embed_dir.joinpath(embed_fname)
            fpaths.append((wav_fpath, embed_fpath))

        # Load Pre-trained embeddings
        # encoder.load_model(encoder_model_fpath)
        # TODO
        # for wav_fpath, embed_fpath in fpaths:
        #   wav = np.load(wav_fpath)
        #   wav = encoder.preprocess_wav(wav)
        #   embed = encoder.embed_utterance(wav)
        #   np.save(embed_fpath, embed, allow_pickle=False)

    def mel_spectrogram(self, wav):
        print(self.preemphasis)
        temp = signal.lfilter([1, -1 * self.preemphasis], [1], wav)
        D = self._stft(temp)

        # Convert to DB
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        D = 20 * np.log10(np.maximum(min_level, D))

        # Normalize
        return np.clip((2 * self.max_abs_value) * ((D - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                       -self.max_abs_value, self.max_abs_value).astype(np.float32)

    def _stft(self, y,):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def pad1d(self,x, max_len, pad_value=0):
        return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

    def pad2d(self, x, max_len, pad_value=0):
        return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)

    def trim_long_silences(self, wav):
        """
        Ensures that segments without voice in the waveform remain no longer than a 
        threshold determined by the VAD parameters in params.py.

        :param wav: the raw waveform as a numpy array of floats 
        :return: the same waveform with silences trimmed away (length <= original wav length)
        """
        # Compute the voice detection window size
        samples_per_window = (self.vad_window_length *
                              self.SAMPLING_RATE) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(
            wav), *(np.round(wav * self.int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.SAMPLING_RATE))
        voice_flags = np.array(voice_flags)

        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate(
                (np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(
            audio_mask, np.ones(self.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]
    def adjust_batch_data(self, batch):
        x_lens = [len(x[0]) for x in batch]
        max_x_len = max(x_lens)

        chars = [self.pad1d(x[0], max_x_len) for x in batch]
        chars = np.stack(chars)

        # Mel spectrogram
        spec_lens = [x[1].shape[-1] for x in batch]
        max_spec_len = max(spec_lens) + 1 
        if max_spec_len % r != 0:
            max_spec_len += r - max_spec_len % r 

        mel_pad_value = -1 * self.max_abs_value
       
        mel = [self.pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
        mel = np.stack(mel)

        # Speaker embedding (SV2TTS)
        embeds = np.array([x[2] for x in batch])

        # Index (for vocoder preprocessing)
        indices = [x[3] for x in batch]


        # Convert all to tensor
        chars = torch.tensor(chars).long()
        mel = torch.tensor(mel)
        embeds = torch.tensor(embeds)

        return chars, mel, embeds, indices
    def load_model(self,model_fpath):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        """
        self.device = torch.device("cpu")
        self._model = Tacotron(embed_dims=self.tts_embed_dims,
                               num_chars=len(symbols),
                               encoder_dims=self.tts_encoder_dims,
                               decoder_dims=self.tts_decoder_dims,
                               n_mels=self.num_mels,
                               fft_bins=self.num_mels,
                               postnet_dims=self.tts_postnet_dims,
                               encoder_K=self.tts_encoder_K,
                               lstm_dims=self.tts_lstm_dims,
                               postnet_K=self.tts_postnet_K,
                               num_highways=self.tts_num_highways,
                               dropout=self.tts_dropout,
                               stop_threshold=self.tts_stop_threshold,
                               speaker_embedding_size=self.speaker_embedding_size).to(self.device)

        self._model.load(model_fpath)
        self._model.eval()
    
    def synthesize_spectrograms(self, texts,
                                embeddings,
                                return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param return_alignments: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """

        # Preprocess text inputs
        inputs = [text_to_int(text.strip()) for text in texts]
        
        # Batch inputs
        batched_inputs = [inputs[i:i+self.synthesis_batch_size]
                             for i in range(0, len(inputs), self.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i+self.synthesis_batch_size]
                             for i in range(0, len(embeddings), self.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            
            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [self.pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            _, mels, alignments = self._model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < self.tts_stop_threshold:
                    m = m[:, :-1]
                specs.append(m)

        return (specs, alignments) if return_alignments else specs
    def start_training(self, preprocessor_dir, out_dir):

        # Load the dataset
        dataset = Path(preprocessor_dir)
        mels_dir = dataset.joinpath("mels")
        audio_dir = out_dir.joinpath("audio")
        embed_dir = out_dir.joinpath("embed")

        # Initialize output directories
        out_dir = Path(out_dir)
        model_dir = out_dir.joinpath("model")
        model_dir.mkdir(exist_ok=True)

        weights_fpath = model_dir / f"synthesizer.pt"
        metadata_fpath = dataset.joinpath("train.txt")

        # Load the mels and audio and text
        dataset_tuples = []
        with open(metadata_fpath, "r") as metadata_file:
            for line in metadata_file:
                line = line.strip()
                if not line:
                    continue
                line = line.split("|")
                mel_fpath = mels_dir.joinpath(line[1])
                embed_fpath = embed_dir.joinpath(line[2])
                mel = np.load(mel_fpath).T.astype(np.float32)
                # Load the embed
                embed = np.load(embed_fpath)
                text = line[3]
                dataset_tuples.append((text_to_int(text), mel, embed))

        # Create the model
        device = torch.device("cpu")
        model = Tacotron(embed_dims=self.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=self.tts_encoder_dims,
                         decoder_dims=self.tts_decoder_dims,
                         n_mels=self.num_mels,
                         fft_bins=self.num_mels,
                         postnet_dims=self.tts_postnet_dims,
                         encoder_K=self.tts_encoder_K,
                         lstm_dims=self.tts_lstm_dims,
                         postnet_K=self.tts_postnet_K,
                         num_highways=self.tts_num_highways,
                         dropout=self.tts_dropout,
                         stop_threshold=self.tts_stop_threshold,
                         speaker_embedding_size=self.speaker_embedding_size).to(device)

        # Initialize the optimizer
        optimizer = optim.Adam(model.parameters())

        # Shuffle the dataset
        np.random.shuffle(dataset_tuples)

        # Get Batch of data to train model
        def get_batches(batch_size):
            batches = []
            for i in range(0, len(dataset_tuples) - batch_size, batch_size):
                batch_tuples = dataset_tuples[i:i+batch_size]
                batches.append(batch_tuples)

            return batches

        # Iterate over the batches
        index = 0
        for batch in get_batches(self.batch_size):
            for texts, mels, embeds in self.adjust_batch_data(batch):
                index += 1
                
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(idx):
                    stop[j, :int(dataset.metadata[k][4])-1] = 0
                
                texts = texts.to(device)
                mels = mels.to(device)
                embeds = embeds.to(device)
                stop = stop.to(device)

                # Forward Pass
                m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

                # Compute loss
                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = m1_loss + m2_loss + stop_loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                # Backup or save model as appropriate
                if index%100 == 0 :
                    backup_fpath = weights_fpath.parent / f"synthesizer.pt"
                    model.save(backup_fpath, optimizer)
