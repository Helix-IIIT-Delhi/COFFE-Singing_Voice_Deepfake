"""Utility Functions for COFFE"""

import torch
import numpy as np
import librosa
import soundfile as sf


def extract_singing_features(audio_path: str, sr: int = 16000) -> torch.Tensor:
    """Extract audio features from singing voice"""
    waveform, _ = librosa.load(audio_path, sr=sr)
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    return waveform


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file"""
    audio, sr_orig = librosa.load(audio_path, sr=sr)
    return torch.from_numpy(audio).float(), sr


def save_audio(audio: torch.Tensor, path: str, sr: int = 16000):
    """Save audio to file"""
    if audio.is_cuda:
        audio = audio.cpu()
    audio = audio.numpy()
    sf.write(path, audio, sr)


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
