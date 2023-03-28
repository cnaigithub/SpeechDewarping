import numpy as np
from scipy.io.wavfile import read
from scipy import signal

import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path, hparams):
    sampling_rate, data = read(full_path)
    if hparams.use_preemphasis:
        data = signal.lfilter([1, -hparams.preemphasis], [1], data)
    if hparams.use_rescale:
        data = data/np.abs(data).max() * hparams.rescaling_max
    else:
        data = data/hparams.max_wav_value
        
    if hparams.use_bshall_mel:
        return data.astype(np.float32), sampling_rate
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
