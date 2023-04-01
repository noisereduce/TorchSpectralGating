import soundfile as sf
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import torch_gating as tg

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAVE_WAV = False
SHOW_PLOT = True
NONSTATIONARY = True
VMIN = -80


def load_audio_files(path: str):
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


if __name__ == '__main__':



    files, x, fs = load_audio_files('data')

    # Initialize the SpectralGate module and apply it to the input data
    tg = tg.TorchSpectralGate(sr=fs, nonstationary=NONSTATIONARY).to(DEVICE)
    y = tg(torch.from_numpy(x).to(DEVICE)).cpu().numpy()

    # Save processed audio
    if SAVE_WAV:
        for i in range(len(files)):
            sf.write(fr"output\{'non-stationary' if NONSTATIONARY else 'stationary'}\{files[i]}",
                     y[i] / np.max(np.abs(y[i])), fs)

    # Display input and output spectrograms as plots
    if SHOW_PLOT:
        idx = 4
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(x[idx] / (np.abs(x[idx])).max())
        axs[0, 1].specgram(x[idx], Fs=fs, cmap='magma', vmin=VMIN)
        axs[1, 0].plot(y[idx] / (np.abs(y[idx])).max())
        axs[1, 1].specgram(y[idx], Fs=fs, cmap='magma', vmin=VMIN)
        fig.suptitle(f"{files[idx]} | {'Non-Stationary' if NONSTATIONARY else 'Stationary'}")

        plt.show(block=True)
