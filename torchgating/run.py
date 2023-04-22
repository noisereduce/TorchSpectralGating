from typing import Optional, Tuple
import argparse
import os
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import torch
import warnings

from .torchgating import TorchGating as TG
from .version import __version__

warnings.filterwarnings("ignore")
EPS = np.finfo(float).eps


def vprint(msg: str, verbose: bool):
    """
     Utility function to print a message if verbose mode is enabled.

     Arguments:
         msg (str): Message to print.
         verbose (bool): Flag indicating whether verbose mode is enabled.
     """
    if verbose:
        print(msg)


def check_dir(path: str, subdir: Optional[str], verbose: bool) -> str:
    """
    Utility function to check whether a directory exists, and if not, create it.

    Arguments:
        path (str): Path to the directory to check/create.
        subdir (Optional[str]): Optional subdirectory to append to the path.
        verbose (bool): Flag indicating whether verbose mode is enabled.

    Returns:
        str: The path with the subdirectory (if any) appended.
    """
    if subdir is not None:
        path = os.path.join(path, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
        vprint(f'Created {path}', verbose)
    return path


def load_audio_files(path: str, verbose: bool) -> tuple:
    """
    Load audio files from a directory

    Arguments:
        path (str): Path to the directory containing audio files or to an audio file
        verbose (bool): Flag indicating whether verbose mode is enabled.

    Returns:
        tuple: A tuple of audio file names, audio data, and sample rate
    """
    if os.path.isdir(path):
        file_list = [f for f in os.listdir(path)]
        dir_path = path
    else:
        file_list = [os.path.basename(path)]
        dir_path = os.path.dirname(os.path.abspath(path))

    assert len(file_list) > 0
    audio_data, sample_rate = [], 0

    for i, f in enumerate(file_list):
        try:
            audio, sr = sf.read(os.path.join(dir_path, f))
            vprint(f'Load {f}', verbose)

            audio_data.append(audio)
            if i == 0:
                sample_rate = sr

            assert sr == sample_rate

        except Exception as e:
            file_list.remove(f)
            raise Warning(f'Could not read {f}. {e}')

    assert len(file_list) > 0
    audio_data = np.vstack(audio_data)

    return file_list, audio_data, sample_rate


def plot_waveform_specgram(x: np.ndarray, y: np.ndarray, fs: int, title: str,
                           vmin: Optional[int] = None, vmax: Optional[int] = None,
                           cmap: str = 'magma', figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot the waveform and spectrogram of input and output audio signals.

    Arguments:
        x (np.ndarray): The input audio signal.
        y (np.ndarray): The output audio signal.
        fs (int): The sampling rate of the audio signals.
        title (str): The title of the plot.
        vmin (Optional[int]): The minimum value for the color scale.
        vmax (Optional[int]): The maximum value for the color scale.
        cmap (str): The name of the colormap to use.
        figsize (Tuple[int, int]): The size of the figure in inches.

    Returns:
        A tuple containing the Matplotlib figure and the axes array.
    """

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, constrained_layout=True)

    axs[0, 0].plot(np.arange(len(x)) / fs, x)
    axs[0, 1].specgram(x, Fs=fs, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1, 0].plot(np.arange(len(y)) / fs, y)
    axs[1, 1].specgram(y, Fs=fs, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.suptitle(title)
    return fig, axs


def parse_args() -> argparse.Namespace:
    """
    Command line interface for torchgating

    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Audio processing script.")
    parser.add_argument('input', type=str,
                        help='Path to a directory containing audio files or to a single audio file.')
    parser.add_argument("-v", "--version", action="version", version=f"torch-gating version: v{__version__}",
                        help='Print torch-gating version')
    parser.add_argument('--output', type=str, default='output',
                        help='Path to a directory to save the processed audio files (default: output).')
    parser.add_argument('--nonstationary', action='store_true',
                        help='Whether to use non-stationary or stationary masking (default: stationary).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information about the processing (default: False).')
    parser.add_argument('--norm', action='store_true',
                        help='Whether to normalize the signals (default: False).')
    parser.add_argument('--graphs', action='store_true',
                        help='Whether to save graphs of the processed audio signals (default: False).')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU for processing (default: False).')
    parser.add_argument('--subdirs', action='store_true',
                        help='Create subdirectories for the processed audio files based on stationary/non-stationary '
                             '(default: False).')
    parser.add_argument('--figsize', type=tuple, default=(10, 6),
                        help='Size of the figure for the displayed spectrograms (default: (10, 6)).')
    parser.add_argument('--figformat', type=str, default='png',
                        help='If figformat is set, it determines the output format (default: png).')
    parser.add_argument('--vmin', type=Optional[int], default=-80,
                        help='Minimum value for the color scale of the spectrograms (default: -80).')
    parser.add_argument('--vmax', type=Optional[int], default=None,
                        help='Maximum value for the color scale of the spectrograms (default: None).')
    parser.add_argument('--cmap', type=str, default='magma',
                        help='Name of the colormap to use for the spectrograms (default: magma).')

    return parser.parse_args()


def main():
    opt = parse_args()

    # Device to run the model on
    device = torch.device("cpu") if opt.cpu else torch.device("cuda")
    assert not opt.cpu or torch.cuda.is_available()

    # Load audio files
    files, x, fs = load_audio_files(opt.input, opt.verbose)
    if opt.norm:
        x /= (np.expand_dims(np.abs(x).max(axis=1), 1) + EPS)

    #  and apply it to the input data
    tg = TG(sr=fs, nonstationary=opt.nonstationary).to(device)
    y = tg(torch.from_numpy(x).to(device)).cpu().numpy()

    if opt.norm:
        y /= (np.expand_dims(np.abs(y).max(axis=1), 1) + EPS)

    subdirs = fr"{'non-stationary' if opt.nonstationary else 'stationary'}" if opt.subdirs else None
    output_dir = check_dir(opt.output, subdirs, opt.verbose)

    for i, filename in enumerate(files):
        # Save processed audio
        outpath = os.path.join(output_dir, filename)
        sf.write(outpath, y[i], fs)
        vprint(fr'Saved {outpath}', opt.verbose)

        if opt.graphs:
            # Display input and output spectrograms as plots
            fig, axs = plot_waveform_specgram(x[i],
                                              y[i],
                                              fs=fs,
                                              cmap=opt.cmap,
                                              vmin=opt.vmin,
                                              vmax=opt.vmax,
                                              figsize=opt.figsize,
                                              title=f"{files[i]} | {'Non-Stationary' if opt.nonstationary else 'Stationary'}")

            outpath = os.path.join(output_dir, f"{filename[:filename.rindex('.')]}.{opt.figformat}")
            fig.savefig(outpath)
            vprint(fr'Saved {outpath}', opt.verbose)

    vprint('Done', opt.verbose)


if __name__ == '__main__':
    main()
