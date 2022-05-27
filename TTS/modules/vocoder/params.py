sample_rate = 16000
n_fft = 800
num_mels = 80
hop_length = 200                             # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
win_length = 800                             # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
fmin = 55
min_level_db = -100
ref_level_db = 20
mel_max_abs_value = 4.                         # Gradient explodes if too big premature convergence if too small.
preemphasis = 0.97                         # Filter coefficient to use if preemphasize is True
apply_preemphasis = True
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode
                                    # below


# WAVERNN / VOCODER --------------------------------------------------------------------------------
voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from 
# mixture of logistics)
voc_upsample_factors = (5, 5, 8)    # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 100
voc_lr = 1e-4
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider 
                                    # than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 8000                   # target number of samples to be generated in each batch entry
voc_overlap = 400                   # number of samples for crossfading between batches
