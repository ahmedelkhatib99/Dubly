
import gc
from modules.vocoder.WaveRNN import WaveRNN
import modules.vocoder.params as hp
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class Vocoder:
    def delete_model_from_memory(self):
        # self._model = None
        self._model.to('cpu')
        del self._model
        torch.cuda.empty_cache()
        # gc.collect()
    def load_model(self, weights_fpath, verbose=True):

        self._model = WaveRNN(
            rnn_dims=hp.voc_rnn_dims,
            fc_dims=hp.voc_fc_dims,
            bits=hp.bits,
            pad=hp.voc_pad,
            upsample_factors=hp.voc_upsample_factors,
            feat_dims=hp.num_mels,
            compute_dims=hp.voc_compute_dims,
            res_out_dims=hp.voc_res_out_dims,
            res_blocks=hp.voc_res_blocks,
            hop_length=hp.hop_length,
            sample_rate=hp.sample_rate,
            mode=hp.voc_mode
        )

        if torch.cuda.is_available():
            self._model = self._model.cuda()
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        checkpoint = torch.load(weights_fpath, self._device)
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()

    def is_loaded(self):
        return self._model is not None

    def infer_waveform(self, mel, normalize=True,  batched=True, target=8000, overlap=800,
                       progress_callback=None):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
        that of the synthesizer!)

        :param normalize:  
        :param batched: 
        :param target: 
        :param overlap: 
        :return: 
        """
        if self._model is None:
            raise Exception("Please load Wave-RNN in memory before using it")

        if normalize:
            mel = mel / hp.mel_max_abs_value
        mel = torch.from_numpy(mel[None, ...])
        wav = self._model.generate(
            mel, batched, target, overlap, hp.mu_law, progress_callback)
        return wav
