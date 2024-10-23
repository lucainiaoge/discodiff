import numpy as np

import torch
import torchaudio

import librosa
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from typing import Type, Optional, Tuple, Union, List

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def set_device_accelerator(force_cpu: bool = False):
    if not force_cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)
        return device, accelerator
    else:
        return "cpu", "cpu"

# from https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

# Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity
def get_velocity(scheduler, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=sample.dtype).to(device=sample.device)
    timesteps = timesteps.to(sample.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(sample.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity

def pad_last_dim(tensor, max_len, pad_val = 0):
    pad_shape = list(tensor.shape)
    assert max_len >= pad_shape[-1]
    pad_shape[-1] = max_len - pad_shape[-1]
    pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype) + pad_val
    return np.concatenate([tensor, pad_tensor], axis=-1)

"""
    cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html
"""
def spectrogram_image(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)

def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 80

    mel_spectrogram_op = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, 
        n_fft=n_fft, 
        win_length=win_length, 
        hop_length=hop_length, 
        center=True, 
        pad_mode="reflect", 
        power=power, 
        norm='slaney', 
        n_mels=n_mels, 
        mel_scale="htk",
    )

    melspec = mel_spectrogram_op(waveform.float().cpu())
    melspec = melspec[0] # TODO: only left channel for now
    return spectrogram_image(melspec, title="MelSpectrogram", ylabel='mel bins (log freq)')