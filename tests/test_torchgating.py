import unittest
import torch
from torchgating import TorchGating as TG


class TestTorchGating(unittest.TestCase):
    """
    Test cases for the TorchGating class.
    """

    def test_nonstationary(self, sr: int = 8000):
        """
        Test Non-Stationary

        Args:
            sr (int): Signal sampling frequency.
        """

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create TorchGating instance
        tg = TG(sr=sr, nonstationary=True).to(device)

        # Apply Spectral Gate to noisy speech signal
        noisy_speech = torch.randn(3, 32000, device=device)
        enhanced_speech = tg(noisy_speech)
        self.assertIsNotNone(enhanced_speech)

    def test_stationary(self, sr: int = 8000):
        """
        Test Stationary.

        Args:
            sr (int): Signal sampling frequency.
        """

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create TorchGating instance
        tg = TG(sr=sr, nonstationary=False).to(device)

        # Apply Spectral Gate to noisy speech signal
        noisy_speech = torch.randn(3, 32000, device=device)
        enhanced_speech = tg(noisy_speech)
        self.assertIsNotNone(enhanced_speech)


if __name__ == '__main__':
    unittest.main()
