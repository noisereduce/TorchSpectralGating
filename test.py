import soundfile as sf
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from typing import Union

from torch_gating import TorchGating as TG

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VMIN = -80
VMAX = None
CMAP = 'magma'
NORM = True
FIGSIZE = (10, 6)


def load_audio_files(path: str) -> tuple:
    """Load audio files from a directory"""
    audio_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    audio_data = []
    sample_rate = 0

    for audio_file in audio_files:
        audio, sr = sf.read(os.path.join(path, audio_file))
        audio_data.append(audio)
        sample_rate = sr

    audio_data = np.vstack(audio_data)

    return audio_files, audio_data, sample_rate


def plot_waveform_specgram(x: np.ndarray, y: np.ndarray, fs: int, title: str, vmin: Union[int, None] = VMIN,
                           vmax: Union[int, None] = VMAX, cmap: str = CMAP, norm: bool = NORM,
                           figsize: tuple = FIGSIZE):
    if norm:
        x /= (np.abs(x)).max()
        y /= (np.abs(y)).max()

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0, 0].plot(x)
    axs[0, 1].specgram(x, Fs=fs, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1, 0].plot(y)
    axs[1, 1].specgram(y, Fs=fs, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.suptitle(title)

    plt.show(block=True)


if __name__ == '__main__':

    files, x, fs = load_audio_files('data')

    # Initialize the SpectralGate module and apply it to the input data
    nonstationary = True
    tg = TG(sr=fs, nonstationary=nonstationary).to(DEVICE)
    y = tg(torch.from_numpy(x).to(DEVICE)).cpu().numpy()

    # Save processed audio
    for i in range(len(files)):
        sf.write(fr"output\{'non-stationary' if nonstationary else 'stationary'}\{files[i]}",
                 y[i] / np.max(np.abs(y[i])), fs)

    # Display input and output spectrograms as plots
    idx = 4
    plot_waveform_specgram(x[idx], y[idx], fs=fs,
                           title=f"{files[idx]} | {'Non-Stationary' if nonstationary else 'Stationary'}")
    plt.show(block=True)
