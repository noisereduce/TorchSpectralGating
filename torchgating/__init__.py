"""
TorchGating is a PyTorch-based implementation of Spectral Gating, an algorithm for denoising audio signals
================================================
Documentation is available in the docstrings and
online at https://github.com/nuniz/TorchSpectralGating/blob/main/README.md.

Contents
--------
torchgating imports all the functions from PyTorch, and in addition provides:
 TorchGating       --- A PyTorch module that applies a spectral gate to an input signal

The "run.py" script provides a command-line interface for applying the SpectralGate algorithm to audio files.
 CLI command       --- torchgating input_path

Public API in the main TorchGating namespace
--------------------------------------
::
 __version__       --- TorchGating version string

References
--------------------------------------
The algorithm was originally proposed by Sainburg et al [1] and was previously implemented in a GitHub repository [2]

[1] Sainburg, Tim, and Timothy Q. Gentner. “Toward a Computational Neuroethology of Vocal Communication:
From Bioacoustics to Neurophysiology, Emerging Tools and Future Directions.”

[2] Sainburg, T. (2019). noise-reduction. GitHub. Retrieved from https://github.com/timsainb/noisereduce.
"""


from .torchgating import TorchGating
from .version import __version__
